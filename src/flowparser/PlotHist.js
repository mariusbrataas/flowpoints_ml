import React, { Component } from 'react';

export function getPlotHist() {
  var msg = '\n    def plot_hist(self):'
  msg += "\n        # Adding plots"
  msg += "\n        plt.plot(self.train_loss_hist, color='blue', label='Training loss')"
  msg += "\n        plt.plot(self.valid_loss_hist, color='springgreen', label='Validation loss')"
  msg += "\n        # Axis labels"
  msg += "\n        plt.xlabel('Epoch')"
  msg += "\n        plt.ylabel('Loss')"
  msg += "\n        plt.legend(loc='upper right')"
  msg += "\n        # Displaying plot"
  msg += "\n        plt.show()"
  return msg
}
