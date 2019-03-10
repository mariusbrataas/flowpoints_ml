import React, { Component } from 'react';

import './Sidebar.css';
import { PyTorchParser } from './parsers/Parsers.js';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { github, atelierForestDark } from 'react-syntax-highlighter/dist/styles/hljs';

import Typography from '@material-ui/core/Typography';
import TabContainer from './TabContainer.js'
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import FormControl from '@material-ui/core/FormControl';
import Drawer from '@material-ui/core/Drawer';
import Button from '@material-ui/core/Button';

import DeleteIcon from '@material-ui/icons/Delete';
import red from '@material-ui/core/colors/red';

import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';

import Tooltip from '@material-ui/core/Tooltip';
import Fab from '@material-ui/core/Fab';
import IconButton from '@material-ui/core/IconButton';
import AddIcon from '@material-ui/icons/Add';

import { MuiThemeProvider, createMuiTheme } from '@material-ui/core/styles';
import OutlinedInput from '@material-ui/core/OutlinedInput';

import { FaGithub, FaLinkedin, FaNpm } from "react-icons/fa";

var htmlToImage = require('html-to-image');

const darktheme = createMuiTheme({
  palette: {
    type: 'dark',
  },
  typography: { useNextVariants: true },
});

const lighttheme = createMuiTheme({
  palette: {
    type: 'light',
  },
  typography: { useNextVariants: true },
});

github.hljs.background = '#ffffff';

export const themes = [
  'red',
  'pink',,
  'purple',
  'deep-purple',
  'indigo',
  'blue',
  'light-blue',
  'green',
  'light-green',
  'lime',
  'yellow',
  'amber',
  'orange',
  'deep-orange',
  'brown',
  'grey',
  'blue-grey',
  'black',
  'white'
]


const SettingsTab = (props) => {
  var env = props.environment;
  return (
    <div style={{ padding:'15px' }}>

      <div>
        <h3 style={{marginTop:'0px'}}>Environment</h3>
        <FormControl>
          <InputLabel htmlFor='libSelect'>Library</InputLabel>
          <Select
            value={env.library}
            inputProps={{ name:'library select', id:'libSelect'}}
            onChange={(e) => {
              var environment = props.refresh().environment;
              environment.library = e.target.value;
              props.updateEnv(environment)
            }}>
            {
              env.libraries.map(libname => {
                return (
                  <MenuItem value={libname}>{libname}</MenuItem>
                )
              })
            }
          </Select>
        </FormControl>
      </div>

      <div style={{paddingTop:'20px'}}>
        <h3>Customize view</h3>

        <div style={{paddingBottom:10}}>
          <FormControl style={{width:'48%', paddingRight:10, paddingBottom:10}}>
            <InputLabel htmlFor='themeselect'>Theme</InputLabel>
            <Select
              value={props.settings.theme}
              inputProps={{ name:'theme select', id:'themeselect'}}
              onChange={(e) => {
                var settings = props.refresh().settings;
                settings.theme = e.target.value;
                props.updateSettings(settings)
              }}>
              {
                themes.map(themename => {
                  return (
                    <MenuItem value={themename}>{themename}</MenuItem>
                  )
                })
              }
            </Select>
          </FormControl>

          <FormControl style={{width:'48%'}}>
            <InputLabel htmlFor='varselect'>Variant</InputLabel>
            <Select
              value={props.settings.variant}
              inputProps={{ name:'vairant select', id:'varselect'}}
              onChange={(e) => {
                var settings = props.refresh().settings;
                settings.variant = e.target.value;
                props.updateSettings(settings)
              }}>
              {
                ['paper','outlined','filled'].map(varname => {
                  return (
                    <MenuItem value={varname}>{varname}</MenuItem>
                  )
                })
              }
            </Select>
          </FormControl>
        </div>

        {
          [
            {r:'darktheme', t:'Dark theme'},
            {r:'showName', t:'Show names'},
            {r:'showShape', t:'Show shapes'},
            {r:'avoidCollisions', t:'Avoid collisions'}
          ].map(tmp => {
            return (
              <FormControlLabel
                style={{paddingTop:0}}
                control={
                  <Switch
                    checked={props.settings[tmp.r]}
                    color='primary'
                    onChange={e => {
                      var settings = props.refresh().settings;
                      settings[tmp.r] = !settings[tmp.r];
                      props.updateSettings(settings)
                    }}/>
                }
                label={tmp.t}/>
            )
          })
        }
      </div>

      <div style={{paddingTop:20}}>
        <Button
          variant="outlined"
          onClick={(e) => {
            htmlToImage.toPng(props.diagramRef).then(function (dataUrl) {
              var img = new Image();
              img.src = dataUrl;
              var link = document.createElement('a');
              link.download = 'diagram.png';
              link.href = dataUrl;
              link.click();
            })
          }}>
          Export png
        </Button>
      </div>

    </div>
  )
}



const CodeTab = props => {

  // Selecting parser
  var parser = null
  if (props.environment.library === 'PyTorch') parser = PyTorchParser;

  // Selecting theme
  const codetheme = props.darktheme ? atelierForestDark : github;

  return (
    <div style={{fontSize:12}}>
      <SyntaxHighlighter language='python' style={codetheme} showLineNumbers>
        {
          props.code
        }
      </SyntaxHighlighter>
    </div>
  )

}



const FlowpointTab = props => {

  var point = props.flowpoints[props.selected];

  if (point) {

    // Helpers
    const params = point.specs.params;
    var fields = { 'int':[], 'double':[], 'bool':[], 'tuple':[], 'select':[] };

    // Input?
    if (point.specs.title === 'Input') {
      fields.int.push(
        <TextField
          label='N dimensions'
          type='number'
          variant='outlined'
          margin='dense'
          style={{width:'155px', paddingRight:'5px'}}
          value={params.n_dims.current}
          onChange={e => {
            var flowpoints = props.refresh().flowpoints;
            var p = flowpoints[props.selected];
            var params = p.specs.params;
            params.n_dims.current = Math.max(Math.min(e.target.value, params.n_dims.max), params.n_dims.min);
            if (params.n_dims.current < params.dimensions.current.length) {
              params.dimensions.current = params.dimensions.current.splice(0, params.n_dims.current)
            } else if (params.n_dims.current > params.dimensions.current.length) {
              Array.from(Array(params.n_dims.current - params.dimensions.current.length).keys()).map(idx => {
                params.dimensions.current.push(1)
              })
            }
            props.updatePointSettings(props.selected, p)
          }}/>
      )
    }

    Object.keys(params).map(paramkey => {
      const param = params[paramkey]
      switch(param.type) {
        case 'int':
          fields.int.push(
            <TextField
              label={paramkey}
              type='number'
              variant='outlined'
              margin='dense'
              style={{width:'155px', paddingRight:'5px'}}
              value={params[paramkey].current}
              onChange={e => {
                var flowpoints = props.refresh().flowpoints;
                var p = flowpoints[props.selected];
                var param = p.specs.params[paramkey];
                param.current = e.target.value === '' ? '' : Math.max(Math.min(e.target.value, param.max), param.min);
                props.updatePointSettings(props.selected, p)
              }}/>
          )
          break;
        case 'double':
          fields.double.push(
            <TextField
              label={paramkey}
              type='number'
              variant='outlined'
              margin='dense'
              style={{width:'155px', paddingRight:'5px'}}
              value={params[paramkey].current}
              onChange={e => {
                var flowpoints = props.refresh().flowpoints;
                var p = flowpoints[props.selected];
                var param = p.specs.params[paramkey];
                param.current = e.target.value === '' ? '' : Math.max(Math.min(e.target.value, param.max), param.min);
                props.updatePointSettings(props.selected, p)
              }}/>
          )
          break;
        case 'bool':
          fields.bool.push(
            <FormControlLabel
              control={
                <Switch
                  checked={param.current === 'True'}
                  color='primary'
                  onChange={e => {
                    var flowpoints = props.refresh().flowpoints;
                    var p = flowpoints[props.selected];
                    var param = p.specs.params[paramkey];
                    if (param.current === 'True') {
                      param.current = 'False';
                    } else {
                      param.current = 'True';
                    }
                    props.updatePointSettings(props.selected, p)
                  }}/>
              }
              label={paramkey}/>
          )
          break;
        case 'tuple':
          fields.tuple.push(
            <div style={{paddingTop:'15px'}}>
              <div><h5 style={{margin:'0px'}}>{paramkey}</h5></div>
              {
                param.current.map((val, index) => {
                  return (
                    <TextField
                      label={paramkey + ' ' + index}
                      type='number'
                      variant='outlined'
                      margin='dense'
                      style={{width:'155px', paddingRight:'5px'}}
                      value={val}
                      onChange={e => {
                        var flowpoints = props.refresh().flowpoints;
                        var p = flowpoints[props.selected]
                        var param = flowpoints[props.selected].specs.params[paramkey];
                        param.current[index] = e.target.value === '' ? '' : Math.max(Math.min(e.target.value, param.max), param.min);
                        props.updatePointSettings(props.selected, p)
                      }}/>
                  )
                })
              }
            </div>
          )
          break;
        case 'select':
          fields.select.push(
            <FormControl>
              <InputLabel htmlFor="selectfield">{paramkey}</InputLabel>
              <Select
                value={params[paramkey].options.indexOf(params[paramkey].current)}
                style={{width:'155px', paddingRight:'5px'}}
                onChange={(e) => {
                  var flowpoints = props.refresh().flowpoints;
                  var p = flowpoints[props.selected];
                  var param = p.specs.params[paramkey];
                  param.current = param.options[e.target.value]
                  props.updatePointSettings(props.selected, p)
                }}
                inputProps={{id:'selectfield'}}>
                {
                  params[paramkey].options.map((opt_val, opt_idx) => {
                    return (
                      <MenuItem value={opt_idx}>{opt_val}</MenuItem>
                    )
                  })
                }
              </Select>
            </FormControl>
          )
          break;
      }
    })

    // List of module types
    const modules = props.environment.getModules()
    var title2key = {}
    Object.keys(modules).map(mod_key => {
      title2key[modules[mod_key].title] = mod_key;
    })

    return (
      <div style={{padding:'15px'}}>

        <div>

          <h2 style={{marginTop:'0px'}}>Flowpoint settings</h2>

          <FormControl style={{width:'100%'}}>
            <Select
              value={point.specs.title}
              inputProps={{ name:'spec select', id:'specselect'}}
              onChange={(e) => {
                if (e.target.value in title2key) {
                  var flowpoints = props.refresh().flowpoints;
                  var p = flowpoints[props.selected];
                  p.specs = modules[title2key[e.target.value]];
                  props.updatePointSettings(props.selected, p)
                }
              }}>
              {
                Object.keys(title2key).map(varname => {
                  return (
                    <MenuItem value={varname}>{varname}</MenuItem>
                  )
                })
              }
            </Select>
          </FormControl>

          <TextField
            label='Name'
            value={point.name}
            margin='tight'
            style={{width:'80%'}}
            onChange={e => {
              var flowpoints = props.refresh().flowpoints;
              var p = flowpoints[props.selected];
              p.name = e.target.value;
              props.updatePointSettings(props.selected, p)
            }}/>

          <IconButton
            aria-label="Delete"
            color={red[500]}
            style={{
              marginLeft:10,
              marginTop:5,
              color:red[300]
            }}
            onClick={props.deleteSelected}>
            <DeleteIcon />
          </IconButton>

        </div>

        <div style={{paddingTop:'25px'}}>
          <h3>Parameters</h3>
          <div>{ fields.int    }</div>
          <div>{ fields.double }</div>
          <div>{ fields.select }</div>
          <div>{ fields.tuple  }</div>
          <div>{ fields.bool   }</div>
        </div>

      </div>
    )
  } else {
    return (
      <div style={{display:'table', width:'100%', height:'50px'}}>
        <div style={{display:'table-cell', verticalAlign:'middle', textAlign:'center'}}>
          Click on a flowpoint to display it's settings.
        </div>
      </div>
    )
  }
}


export const Sidebar = (props) => {
  return (
    <MuiThemeProvider theme={props.darktheme ? darktheme : lighttheme}>

      <Drawer
        variant='persistent'
        anchor='left'
        open={props.open}>

        <div
          class='fullSidebar'
          style={{
            maxWidth:props.settings.drawerWidth,
            color: (props.darktheme ? 'white' : 'black')
          }}>

          <Typography gutterBottom variant="h5" component="h2" style={{padding:'15px'}}>
            Flowpoints ML
          </Typography>

          <div style={{position:'absolute', right:5, top:5}}>
            <IconButton target='_blank' href='https://www.npmjs.com/package/flowpoints'>
              <FaNpm/>
            </IconButton>
            <IconButton target='_blank' href='https://www.linkedin.com/in/marius-brataas-355567106/'>
              <FaLinkedin/>
            </IconButton>
            <IconButton target='_blank' href='https://github.com/mariusbrataas/flowpoints_ml#readme'>
              <FaGithub/>
            </IconButton>
          </div>

          <div>
            <TabContainer
              tabs={['Misc', 'Code', 'Flowpoint']}
              tab={props.settings.tab}
              width={350}
              onSelectTab={(value) => {
                var settings = props.refresh().settings;
                settings.tab = value;
                if (value === 'Code') {
                  settings.drawerWidth = 600
                } else {
                  settings.drawerWidth = 350
                }
                props.updateSettings(settings);
              }}/>
          </div>

          <div class='scrollBox' style={{backgroundColor:(props.darktheme ? atelierForestDark.hljs.background : null)}}>

            <div>
              {
                props.settings.tab === 'Misc' ? <SettingsTab
                  environment={props.environment}
                  flowpoints={props.flowpoints}
                  refresh={props.refresh}
                  updateEnv={props.updateEnv}
                  settings={props.settings}
                  updateSettings={props.updateSettings}
                  diagramRef={props.diagramRef}/> : null
              }
              {
                props.settings.tab === 'Code' ? <CodeTab
                  environment={props.environment}
                  flowpoints={props.flowpoints}
                  refresh={props.refresh}
                  code={props.code}
                  darktheme={props.darktheme}/> : null
              }
              {
                props.settings.tab === 'Flowpoint' ? <FlowpointTab
                  selected={props.selected}
                  settings={props.settings}
                  environment={props.environment}
                  flowpoints={props.flowpoints}
                  refresh={props.refresh}
                  deleteSelected={props.deleteSelected}
                  updatePointSettings={props.updatePointSettings}/> : null
              }
            </div>

          </div>

        </div>

      </Drawer>

    </MuiThemeProvider>
  )
}
