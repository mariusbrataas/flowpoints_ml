import React from 'react';

import { getFlowCodeName } from './CommonTools'

export function getFit() {
  var msg = '\n    def fit(self, train_loader, validation_loader=None, epochs=10, show_progress=True):'
  msg += '\n        # Possibly prepping progress message'
  msg += '\n        if show_progress:'
  msg += '\n            epoch_l = max(len(str(epochs)), 2)'
  msg += "\n            print('Training model...')"
  msg += "\n            print('%sEpoch   Training loss   Validation loss   Duration' % ''.rjust(2 * epoch_l - 4, ' '))"
  msg += '\n            t = time.time()'
  msg += '\n        # Looping through epochs'
  msg += '\n        for epoch in range(epochs):'
  msg += '\n            self.fit_step(train_loader)                 # Optimizing weights'
  msg += '\n            if validation_loader != None:               # Do validation?'
  msg += '\n                self.validation_step(validation_loader) # Calculating validation loss'
  msg += '\n            # Possibly printing progress'
  msg += '\n            if show_progress:'
  msg += "\n                print('%s/%s' % (str(epoch + 1).rjust(epoch_l, ' '), str(epochs).ljust(epoch_l, ' ')),"
  msg += "\n                    '| %s' % str(round(self.train_loss_hist[-1], 8)).ljust(13, ' '),"
  msg += "\n                    '| %s' % str(round(self.valid_loss_hist[-1], 8)).ljust(15, ' '),"
  msg += "\n                    '| %ss' % str(round(time.time() - t, 3)).rjust(7, ' '))"
  msg += '\n                t = time.time()'
  return msg
}
