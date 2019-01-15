import React from 'react';

import { getInitials } from './Initials'
import { getConstructor } from './Constructor'
import { getOrdered, getInputs } from './FlowOrder'
import { getStateNames } from './FlowStateNames'
import { getForward } from './Forward'
import { getFitStep, getValidationStep } from './Step'
import { getFit } from './Fit'
import { getPlotHist } from './PlotHist'
import { getSaveLoad } from './SaveLoad'

export function parseFlowPoints(flowpoints) {
  if (Object.keys(flowpoints).length != 0) {
    var msg = getInitials()
    const inps = getInputs(flowpoints)
    const order = getOrdered(flowpoints, inps)
    const statenames = getStateNames(flowpoints, order)
    msg += '\n' + getConstructor(flowpoints, order, inps, statenames)
    msg += '\n' + getForward(flowpoints, order, inps, statenames)
    msg += '\n' + getFitStep()
    msg += '\n' + getValidationStep()
    msg += '\n' + getFit()
    msg += '\n' + getPlotHist()
    msg += '\n' + getSaveLoad(flowpoints, order)
    msg += '\n\n'
  } else {
    var msg = 'Nothing yet'
  }
  return msg
}
