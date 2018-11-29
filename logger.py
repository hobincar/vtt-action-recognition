# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import tempfile

import moviepy.editor as mpy
import numpy as np
import scipy.misc
from io import BytesIO
import tensorflow as tf


class Logger(object):

    def __init__(self, log_dir, max_queue=10):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir, max_queue=max_queue)

    def text(self, tag, text, step):
        """Log text."""
        text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        self.writer.add_summary(summary, step)

    def scalar(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def gif(self, tag, images, step, fps=5):
        """ Given a 4D numpy tensor of images, log as a gif. """
        with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
        clip = mpy.ImageSequenceClip(list(images), fps=fps)
        clip.write_gif(fname, verbose=False, progress_bar=False)
        with open(fname, 'rb') as f: enc_gif = f.read()
        os.remove(fname)
        # create a tensorflow image summary protobuf:
        thwc = images.shape
        im_summ = tf.Summary.Image()
        im_summ.height = thwc[1]
        im_summ.width = thwc[2]
        im_summ.colorspace = 3 # fix to 3 == RGB
        im_summ.encoded_image_string = enc_gif
        # create a summary obj:
        summary = tf.Summary()
        summary.value.add(tag=tag, image=im_summ)
        # summ_str = summ.SerializeToString()
        self.writer.add_summary(summary, step)

    def histo(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

