import React from 'react';

function getWelcome() {
  var msg = '# Created with https://mariusbrataas.github.io/torchflow/\n'
  return msg
}

function getImports() {
  var msg = '\n# Importing torch tools'
  msg += '\nimport torch'
  msg += '\nfrom torch import nn, optim, cuda'
  msg += '\n\n# Importing other libraries'
  msg += '\nimport numpy as np'
  msg += '\nimport matplotlib.pyplot as plt'
  msg += '\nimport time'
  return msg
}

export function getInitials() {
  return getWelcome() + getImports()
}
