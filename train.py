import json
import os

import cv2
import numpy as np
import tensorflow as tf

from nets import c3d as network
from config import TrainConfig as C
from dataset import load_train_dataset, load_test_dataset
from demo import generate_frame
from logger import Logger


# Basic model parameters
GPU_LIST = [ int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
N_GPU = len(GPU_LIST)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

# A custom logger is introduced to log gifs
summary_writer = Logger(C.log_dpath, max_queue=100)


def placeholder_inputs():
    images_placeholder = tf.placeholder(tf.float32, shape=(
        N_GPU * C.batch_size,
        C.n_frames_per_clip,
        C.crop_size,
        C.crop_size,
        C.n_channels))
    labels_placeholder = tf.placeholder(tf.float32, shape=(N_GPU * C.batch_size, C.n_actions))
    return images_placeholder, labels_placeholder


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(logits, labels):
    cross_entropy_loss = tf.reduce_mean(
        tf.reduce_sum(
            C.class_weights * tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
            axis=1,
        )
    )
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    total_loss = cross_entropy_loss + weight_decay_loss
    return cross_entropy_loss, weight_decay_loss, total_loss


def tower_acc(logits, labels):
    logits = tf.round(tf.nn.sigmoid(logits))
    labels = tf.round(labels)
    correct_pred = tf.equal(logits, labels)

    correct_pred = tf.cast(correct_pred, tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    return accuracy


def score(preds, actuals):
    preds = (preds > 0.5).astype(np.bool)
    actuals = actuals.astype(np.bool)

    TP = np.logical_and(preds, actuals).sum(axis=1)
    FP = np.logical_and(preds, ~actuals).sum(axis=1)
    FN = np.logical_and(~preds, actuals).sum(axis=1)

    precision = TP / (TP + FP)
    precision[np.isnan(precision)] = 0
    precision = precision.mean()
    recall = TP / (TP + FN)
    recall[np.isnan(recall)] = 0
    recall = recall.mean()
    f1score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1score


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def pred_real_to_table(preds, reals, prefix=""):
    lines = [
        [ "real", "pred" ],
    ]
    for i, (pred, real) in enumerate(zip(preds, reals), 1):
        real_label_indices = np.where(real == 1)[0]
        real_label_indices = ", ".join([ str(i) for i in real_label_indices ])
        pred_sigmoid = 1 / (1 + np.exp(-pred))
        pred_label_indices = np.argsort(-pred)[:C.topk]
        pred_label_indices = ", ".join([ "{}({:.2f})".format(i, pred_sigmoid[i]) for i in pred_label_indices ])
        lines.append([ real_label_indices, pred_label_indices ])
    return lines


def clip_summary_with_text(clip, actual, pred):
    actual_labels = np.where(actual == 1)[0]
    ground_truths = [ C.idx2rep[str(a)] for a in actual_labels ]

    predict_scores = np.exp(pred) / sum(np.exp(pred))
    topk_idxs = np.argsort(predict_scores)[-C.topk:]
    topk_actions = [ C.idx2rep[str(idx)] for idx in topk_idxs ]
    topk_scores = predict_scores[topk_idxs]
    actions = [ (action, score) for action, score in zip(topk_actions, topk_scores) ]

    new_clip = []
    for frame in clip:
        new_frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
        new_frame = generate_frame(new_frame, ground_truths, actions, pane_width=1000)
        new_clip.append(new_frame)
    new_clip = np.asarray(new_clip)
    return new_clip


def train_log(clips, preds, gts, step):
    precision, recall, f1score = score(preds, gts)
    pred_summary = pred_real_to_table(preds, gts)
    gif_summary = clip_summary_with_text(clips[0], gts[0], preds[0])
    summary_writer.scalar("train/precision", precision, step)
    summary_writer.scalar("train/recall", recall, step)
    summary_writer.scalar("train/f1score", f1score, step)
    summary_writer.text("train/prediction", pred_summary, step)
    clip_postfix = "{} ~ {}".format(step / C.train_log_every // 10 * 10, step / C.test_log_every // 10 * 10 * 2 - 1)
    summary_writer.gif("train/clip/{}".format(clip_postfix), gif_summary, step)


def test_log(clips, preds, gts, step):
    precision, recall, f1score = score(preds, gts)
    pred_summary = pred_real_to_table(preds, gts)
    gif_summary = clip_summary_with_text(clips[0], gts[0], preds[0])
    summary_writer.scalar("test/precision", precision, step)
    summary_writer.scalar("test/recall", recall, step)
    summary_writer.scalar("test/f1score", f1score, step)
    summary_writer.text("test/prediction", pred_summary, step)
    clip_from = step // C.test_log_every // 10 * C.test_log_every * 10
    clip_to = (step // C.test_log_every // 10 + 1) * C.test_log_every * 10
    summary_writer.gif("test/clip/{} ~ {}".format(clip_from, clip_to), gif_summary, step)


def build_train_model(weights, biases):
    global_step = tf.get_variable(
        'global_step',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False
    )

    images_placeholder, labels_placeholder = placeholder_inputs()
    tower_grads_stable = []
    tower_grads_finetune = []
    logits = []
    opt_stable = tf.train.AdamOptimizer(C.lr_stable)
    opt_finetuning = tf.train.AdamOptimizer(C.lr_finetune)

    for i, gpu_index in enumerate(GPU_LIST):
        with tf.device('/gpu:%d' % gpu_index):
            varlist_finetune = [ weights['out'], biases['out'] ]
            varlist_stable = list( set(list(weights.values()) + list(biases.values())) - set(varlist_finetune) )
            logit, _ = network.inference(
                _X=images_placeholder[i * C.batch_size:(i + 1) * C.batch_size, :, :, :, :],
                _dropout=0.5,
                batch_size=C.batch_size,
                _weights=weights,
                _biases=biases)
            cross_entropy_loss, weight_decay_loss, loss = tower_loss(
                logits=logit,
                labels=labels_placeholder[i * C.batch_size:(i + 1) * C.batch_size])

            grads_stable = opt_stable.compute_gradients(loss, varlist_stable)
            grads_finetune = opt_finetuning.compute_gradients(loss, varlist_finetune)
            tower_grads_stable.append(grads_stable)
            tower_grads_finetune.append(grads_finetune)
            logits.append(logit)
    logits = tf.concat(logits, 0)
    accuracy = tower_acc(logits, labels_placeholder)
    grads_stable = average_gradients(tower_grads_stable)
    grads_finetune = average_gradients(tower_grads_finetune)
    apply_gradient_stable = opt_stable.apply_gradients(grads_stable)
    apply_gradient_finetune = opt_finetuning.apply_gradients(grads_finetune, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(C.moving_average_decay)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_stable, apply_gradient_finetune, variables_averages_op)

    return {
        "varlist_stable": varlist_stable,
        "varlist_finetune": varlist_finetune,
        "images_placeholder": images_placeholder,
        "labels_placeholder": labels_placeholder,
        "logits": logits,
        "accuracy": accuracy,
        "loss": loss,
        "train_op": train_op,
    }


def build_test_model(weights, biases):
    images_placeholder, labels_placeholder = placeholder_inputs()
    logits = []
    for i, gpu_index in enumerate(GPU_LIST):
        with tf.device('/gpu:%d' % gpu_index):
            logit, _ = network.inference(
                _X=images_placeholder[i * C.batch_size:(i + 1) * C.batch_size, :, :, :, :],
                _dropout=1,
                batch_size=C.batch_size,
                _weights=weights,
                _biases=biases)
            cross_entropy_loss, weight_decay_loss, loss = tower_loss(
                logits=logit,
                labels=labels_placeholder[i * C.batch_size:(i + 1) * C.batch_size])

            logits.append(logit)
    logits = tf.concat(logits, 0)
    accuracy = tower_acc(logits, labels_placeholder)

    return {
        "images_placeholder": images_placeholder,
        "labels_placeholder": labels_placeholder,
        "logits": logits,
        "accuracy": accuracy,
        "loss": loss,
    }


def run_training():
    with tf.Graph().as_default():
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'out': _variable_with_weight_decay('wout_finetune', [4096, C.n_actions], 0.0005)
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                'out': _variable_with_weight_decay('bout_finetune', [C.n_actions], 0.000),
            }

        # crop_mean = np.load(C.crop_mean_fpath)
        # crop_mean = crop_mean.reshape([C.n_frames_per_clip, C.crop_size, C.crop_size, 3])

        # Build model
        train_model = build_train_model(weights, biases)
        test_model = build_test_model(weights, biases)


        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(list(weights.values()) + list(biases.values()))

        # Load train dataset
        train_dataset = load_train_dataset(C.train_data_fpath, N_GPU * C.batch_size)
        train_iterator = train_dataset.make_initializable_iterator()
        train_next_batch = train_iterator.get_next()
        sess.run(train_iterator.initializer)

        # Load test dataset
        test_dataset = load_test_dataset(C.test_data_fpath, N_GPU * C.batch_size, shuffle=True, repeat=True)
        test_iterator = test_dataset.make_initializable_iterator()
        test_next_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)

        # Load a pretrained model (if exists)
        if C.use_pretrained_model:
            variables = list(weights.values()) + list(biases.values())
            restorer = tf.train.Saver(train_model["varlist_stable"])
            restorer.restore(sess, C.pretrained_model_fpath)

        # Initialize
        init = tf.global_variables_initializer()
        sess.run(init)

        # Train
        for step in range(1, C.n_iterations + 1):
            train_clips, train_labels = sess.run(train_next_batch)
            sess.run(train_model["train_op"], feed_dict={
                train_model["images_placeholder"]: train_clips,
                train_model["labels_placeholder"]: train_labels,
            })

            # Log train
            if step % C.train_log_every == 0:
                train_clips, train_labels = sess.run(train_next_batch)
                preds, acc, loss_val = sess.run(
                    [train_model["logits"], train_model["accuracy"], train_model["loss"]],
                    feed_dict={
                        train_model["images_placeholder"]: train_clips,
                        train_model["labels_placeholder"]: train_labels,
                    })
                loss_val = np.mean(loss_val)
                print("Train acc.: {:.3f}\tloss: {:.3f}".format(acc, loss_val))
                summary_writer.scalar("train/accuracy", acc, step)
                summary_writer.scalar("train/loss", loss_val, step)

                train_log(train_clips, preds, train_labels, step)

            # Log test
            if step % C.test_log_every == 0:
                test_clips, test_labels, test_frames = sess.run(test_next_batch)
                preds, acc, loss_val = sess.run(
                    [test_model["logits"], test_model["accuracy"], test_model["loss"]],
                    feed_dict={
                        test_model["images_placeholder"]: test_clips,
                        test_model["labels_placeholder"]: test_labels,
                    })
                loss_val = np.mean(loss_val)
                print("Test acc.: {:.3f}\t loss: {:.3f}".format(acc, loss_val))
                summary_writer.scalar("test/accuracy", acc, step)
                summary_writer.scalar("test/loss", loss_val, step)

                test_log(test_clips, preds, test_labels, step)

            # Save a checkpoint
            if step % C.save_every == 0:
                if not os.path.exists(os.path.dirname(C.model_fpath)):
                    os.makedirs(os.path.dirname(C.model_fpath))
                saver.save(sess, C.model_fpath, global_step=step)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

