import React, { Component } from 'react';

import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import Fab from '@material-ui/core/Fab';

import { FaLinkedin, FaFacebook, FaGithub } from 'react-icons/fa';


export const TopBar = (props) => {
  return (
    <AppBar position="fixed" style={{height:'60px', background:'#0a9dff', boxShadow:'none'}}>
        <Toolbar>
          <div style={{flexGrow:1}}>
            <Typography variant="h6" color="inherit">
              Flowpoints ML
            </Typography>
          </div>
          <IconButton
            style={{color:'#ffffff'}}
            href="https://github.com/mariusbrataas"
            target="_blank">
            <FaGithub/>
          </IconButton>
          <IconButton
            style={{color:'#ffffff'}}
            href="https://www.linkedin.com/in/marius-brataas-355567106/"
            target="_blank">
            <FaLinkedin/>
          </IconButton>
          <IconButton
            style={{color:'#ffffff'}}
            href="https://www.facebook.com/marius.brataas"
            target="_blank">
            <FaFacebook/>
          </IconButton>
        </Toolbar>
      </AppBar>
  )
}
