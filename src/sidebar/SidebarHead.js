import React from 'react';

// Material
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';

// Icons
import { FaGithub, FaLinkedin, FaNpm } from "react-icons/fa";
import { Link } from '@material-ui/core';


export const SidebarHead = props => {
  return (
    <div>

      <Typography href='https://mariusbrataas.github.io/flowpoints_ml' gutterBottom variant="h5" component="h2" style={{padding:'15px'}}>
        <Link href='https://mariusbrataas.github.io/flowpoints_ml' color='inherit' underline='none'>
          Flowpoints
        </Link>
      </Typography>

      <div style={{position:'absolute', right:5, top:5}}>
        <IconButton target='_blank' href='https://www.npmjs.com/package/flowpoints'>
          <FaNpm/>
        </IconButton>
        <IconButton target='_blank' href='https://www.linkedin.com/in/mariusbrataas/'>
          <FaLinkedin/>
        </IconButton>
        <IconButton target='_blank' href='https://github.com/mariusbrataas/flowpoints_ml#readme'>
          <FaGithub/>
        </IconButton>
      </div>

    </div>
  )
}