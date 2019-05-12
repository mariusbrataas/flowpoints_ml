import React, { Component } from 'react';
import { Fab, Tooltip } from '@material-ui/core';

import AddIcon from '@material-ui/icons/Add';
import MenuIcon from '@material-ui/icons/Menu';
import FileCopyIcon from '@material-ui/icons/FileCopy';
import LinkIcon from '@material-ui/icons/Link';
import HelpIcon from '@material-ui/icons/Help';


const ButtonContainer = props => {
  return (
    <Tooltip title={props.tooltip} placement="right" disableTriggerFocus disableFocusListener>
      <Fab
        style={{
          background: props.color,
          color: '#ffffff',
          zIndex: 6,
          boxShadow: 'none'
        }}
        aria-label={props.tooltip}
        onClick={() => {
          if (props.onClick) props.onClick()
        }}>
        {
          props.children
        }
      </Fab>
    </Tooltip>
  )
}


export const MainButtons = props => {
  var visual = props.state.visual;
  return (
    <div
      style={{
        bottom:'5px',
        left: visual.drawerWidth * visual.drawerOpen + 5 + 'px',
        position: 'fixed',
        transition: 'left 0.4s ease-out'
      }}>
      <div>

        <div style={{paddingBottom:4}}>
          <ButtonContainer
            color='#00b0ff'
            tooltip='Add flowpoint'
            onClick={props.addFlowpoint}>
            <AddIcon/>
          </ButtonContainer>
        </div>

        <div style={{paddingBottom:4}}>
          <ButtonContainer
            color='#2979ff'
            tooltip='Copy code to clip-board'
            onClick={props.copyCode}>
            <FileCopyIcon/>
          </ButtonContainer>
        </div>

        <div style={{paddingBottom:4}}>
          <ButtonContainer
            color='#3d5afe'
            tooltip='Share link to current model'
            onClick={props.createLink}>
            <LinkIcon/>
          </ButtonContainer>
        </div>

        <div style={{paddingBottom:4}}>
          <ButtonContainer
            color='#651fff'
            tooltip='Help'
            onClick={props.showHideHelp}>
            <HelpIcon/>
          </ButtonContainer>
        </div>

        <div>
          <ButtonContainer
            color='#b0bec5'
            tooltip='Show/hide sidebar'
            onClick={props.showHide}>
            <MenuIcon/>
          </ButtonContainer>
        </div>

      </div>
    </div>
  )
}