import React, { Component } from 'react';
import './App.css';

import { Flowpoint, Flowspace } from 'flowpoints';
import { FlowOrder } from './sidebar/FlowOrder.js';
import { PyTorchParser } from './sidebar/parsers/Parsers.js';
import { PyTorchModules } from './libraries/pytorch.js';
import { Sidebar } from './sidebar/Sidebar.js';
import { parseToQuery, parseFromQuery } from './URLparser.js';
import { postToDB, getDB } from './DBhandler.js';

import Fab from '@material-ui/core/Fab';
import AddIcon from '@material-ui/icons/Add';
import MenuIcon from '@material-ui/icons/Menu';
import FileCopyIcon from '@material-ui/icons/FileCopy';
import LinkIcon from '@material-ui/icons/Link';
import Snackbar from '@material-ui/core/Snackbar';
import SnackbarContent from '@material-ui/core/SnackbarContent';

import green from '@material-ui/core/colors/green';
import red from '@material-ui/core/colors/red';
import grey from '@material-ui/core/colors/grey';
import blue from '@material-ui/core/colors/blue';
import indigo from '@material-ui/core/colors/indigo';
import deepPurple from '@material-ui/core/colors/deepPurple';
import lightBlue from '@material-ui/core/colors/lightBlue';
import teal from '@material-ui/core/colors/teal';
import blueGrey from '@material-ui/core/colors/blueGrey';

import copy from 'copy-to-clipboard';

function shapeBox(shape) {
  var msg = '['
  shape.map(val => {
    msg += val + ','
  })
  if (shape.length > 0) msg = msg.substring(0, msg.length - 1)
  msg += ']'
  return (
    <div style={{textAlign:'center', paddingBottom:'10px'}}>
      {
        msg
      }
    </div>
  )
}

function ReplaceAll(str, search, replacement) {
  var newstr = ''
  str.split(search).map(val => {newstr += val + replacement})
  return newstr.substring(0, newstr.length - replacement.length)
}


class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      lastPos: {x:50, y:-40},
      code: '',
      selected: null,
      count: 0,
      flowpoints: {},
      current_url: '',
      url: '',
      settings:{
        tab: 'Misc',
        theme: 'indigo',
        background: 'black',
        variant:'outlined',
        drawerWidth: 350,
        darktheme: true,
        showShape: false,
        showName: false,
        showSidebar: true,
        avoidCollisions: true,
        snackshow: false,
        snackmsg: 'Hello world',
        snackcolor: green['A700'],
        modelUrl: null
      },
      environment: {
        library: 'PyTorch',
        libraries: [ 'PyTorch' ],
        getModules: PyTorchModules
      }
    }

    // Helpers
    this.baseUrl = window.location.href.split('/?')[0]
    if (this.baseUrl[this.baseUrl.length - 1] !== '/') this.baseUrl += '/'
    this.lib = {}
    this.diagramRef = null;

    // Binding class methods
    this.addFlowpoint = this.addFlowpoint.bind(this);
    this.handleClickPoint = this.handleClickPoint.bind(this);
    this.deleteSelected = this.deleteSelected.bind(this);
    this.prepCode = this.prepCode.bind(this);
    this.prepOutputShapes = this.prepOutputShapes.bind(this);
    this.updatePointSettings = this.updatePointSettings.bind(this);

  }


  componentDidMount() {

    // Creating alert on closing window
    window.onbeforeunload = e => {
      if (this.state.count > 0) return 'Any unsaved data will be lost';
      return null
    }

    // Query?
    var query = window.location.href.split(this.baseUrl)[1]

    if (query.includes('p=')) {

      // Snack message
      var settings = this.state.settings;
      settings.snackshow = true;
      settings.snackmsg = 'Trying to load model';
      settings.snackcolor = blue['A700'];
      settings.modelUrl = this.baseUrl + query.substring(0, 18)
      this.setState({settings})

      try {

        getDB(data => {

          query = query.split('?p=')[1].substring(0, 15)
          query = data[query]

          if (query) {
            query = ReplaceAll(query, 'lll', '.')
            var newlib = parseFromQuery(query)

            // Updating settings and env
            settings = this.state.settings;
            var environment = this.state.environment;
            settings.darktheme = newlib.darktheme;
            settings.showName = newlib.showName;
            settings.theme = newlib.theme;
            settings.background = newlib.background;
            settings.variant = newlib.variant;
            settings.showSidebar = newlib.showSidebar;
            environment.library = newlib.library;

            // Snack message
            settings.snackshow = true;
            settings.snackmsg = 'Loaded model from URL';
            settings.snackcolor = green['A700'];

            // Updating self
            this.setState({
              count:newlib.count,
              flowpoints:newlib.flowpoints,
              settings:settings,
              environment:environment
            }, () => {
              this.prepOutputShapes();
              this.prepCode();
            })

          }

        })

      } catch(err) {
        console.log(err)
        var settings = this.state.settings;
        settings.snackshow = true;
        settings.snackmsg = 'Failed to load from URL';
        settings.snackcolor = red['A400'];
        this.setState({ settings });
      }

    } else {
      this.prepOutputShapes();
      this.prepCode();
    }
  }


  prepCode() {

    // Selecting parser
    var parser = null;
    if (this.state.environment.library === 'PyTorch') parser = PyTorchParser;

    // Parsing code and updating state
    this.setState({ code:parser(this.state.flowpoints, this.lib, this.state.settings.modelUrl) + '\n\n\n' })

  }


  prepOutputShapes() {

    // Helper
    var flowpoints = this.state.flowpoints;

    // Getting floworder and updating lib
    this.lib = FlowOrder(flowpoints)

    // Setting output-shapes of inputs
    var visited = []
    this.lib.order.map(key => {
      if (this.lib.lib[key].specTitle === 'Input') {
        flowpoints[key].output_shape = flowpoints[key].specs.outshape(null, flowpoints[key].specs.params)
        visited.push(key)
      }
    })

    // Setting all output-shapes
    this.lib.order.map(key => {
      if (!visited.includes(key)) {
        var bestInp = null
        Object.keys(this.lib.lib[key].inputs).map(inp_key => {
          if (visited.includes(inp_key)) bestInp = inp_key
        })
        flowpoints[key].specs.params = flowpoints[key].specs.autoparams(flowpoints[bestInp].output_shape, flowpoints[key].specs.params)
        flowpoints[key].output_shape = flowpoints[key].specs.outshape(flowpoints[bestInp].output_shape, flowpoints[key].specs.params)
        visited.push(key)
      }
    })

    // Updating state
    this.setState({ flowpoints })

  }


  updatePointSettings(key, settings) {
    var flowpoints = this.state.flowpoints;
    flowpoints[key] = settings;
    this.setState({ flowpoints });
    this.prepOutputShapes();
    this.prepCode();
  }


  deleteSelected() {
    var flowpoints = this.state.flowpoints;
    delete flowpoints[this.state.selected]
    Object.keys(flowpoints).map(test_key => {
      if (this.state.selected in flowpoints[test_key].outputs) {
        delete flowpoints[test_key].outputs[this.state.selected];
      }
    })
    this.setState({flowpoints, selected:null});
    this.prepOutputShapes();
    this.prepCode();
  }


  addFlowpoint() {
    var flowpoints = this.state.flowpoints;
    flowpoints['' + this.state.count] = {
      name: '',
      output_shape: [],
      pos: {x:this.state.lastPos.x, y:this.state.lastPos.y + 90},
      outputs: {},
      isHover: false,
      specs: this.state.environment.getModules().linear
    }
    if (this.state.selected) {
      flowpoints[this.state.selected].outputs['' + this.state.count] = {
        output:'auto',
        input:'auto'
      }
    }
    this.setState({ flowpoints, count:this.state.count + 1, lastPos:{x:this.state.lastPos.x, y:this.state.lastPos.y + 90}, selected:'' + (this.state.count) })
    this.prepOutputShapes();
    this.prepCode();
  }


  handleClickPoint(key, e) {
    var selected = this.state.selected
    var flowpoints = this.state.flowpoints
    if (e.shiftKey) {
      if (selected === null) {
        selected = key
      } else {
        if (selected !== key) {
          var p1 = flowpoints[selected]
          if (key in p1.outputs) {
            delete p1.outputs[key]
          } else {
            p1.outputs[key] = {
              output:'auto',
              input:'auto',
              onClick:this.handleClickLine
            }
          }
        }
      }
    } else {
      selected = (selected === null ? key : (selected === key ? null : key))
    }
    this.setState({selected, flowpoints})
    this.prepOutputShapes();
    this.prepCode();
  }


  render() {
    return (
      <div style={{backgroundColor: (this.state.settings.darktheme ? 'black' : 'white')}}>

        <Sidebar
          selected={this.state.selected}
          open={this.state.settings.showSidebar}
          flowpoints={this.state.flowpoints}
          settings={this.state.settings}
          environment={this.state.environment}
          refresh={() => {return this.state}}
          updateSettings={(settings) => {this.setState({settings})}}
          updateEnv={(environment) => {this.setState({environment})}}
          deleteSelected={this.deleteSelected}
          darktheme={this.state.settings.darktheme}
          code={this.state.code}
          updatePointSettings={this.updatePointSettings}
          diagramRef={this.diagramRef}/>

        <Flowspace
          theme={this.state.settings.theme}
          variant={this.state.settings.variant}
          background={this.state.settings.darktheme ? 'black' : 'white'}
          selected={this.state.selected}
          getDiagramRef={ref => {this.diagramRef = ref}}
          avoidCollisions={this.state.settings.avoidCollisions}
          onClick={e => {this.setState({ selected:null })}}
          style={{
            height:'100vh',
            width:('calc(100vw - ' + this.state.settings.drawerWidth * this.state.settings.showSidebar + ')'),
            marginLeft:this.state.settings.drawerWidth * this.state.settings.showSidebar + 'px',
            transition:['margin-left 0.4s ease-out','background-color 0.2s ease-out']
          }}>
          {
            Object.keys(this.state.flowpoints).map(key => {
              const point = this.state.flowpoints[key];
              return (
                <Flowpoint
                  key={key}
                  snap={{ x:10, y:10 }}
                  startPosition={point.pos}
                  outputs={point.outputs}
                  onClick={(e) => {
                    this.handleClickPoint(key, e);
                  }}
                  onDrag={pos => {
                    var flowpoints = this.state.flowpoints;
                    flowpoints[key].pos = pos;
                    this.setState({ flowpoints, lastPos:pos })
                  }}
                  style={{
                    height:'auto',
                    maxHeight: this.state.settings.showShape ? 150 : 50
                  }}>
                  <div style={{height:'auto'}}>
                    <div style={{display:'table', width:'100%', height:'50px'}}>
                      <div style={{display:'table-cell', verticalAlign:'middle', textAlign:'center'}}>
                        {
                          this.state.settings.showName ? (point.name !== '' ? point.name : ('p_' + key)) : point.specs.title
                        }
                      </div>
                    </div>
                    {
                      this.state.settings.showShape ? shapeBox(point.output_shape) : null
                    }
                  </div>
                </Flowpoint>
              )
            })
          }
        </Flowspace>

        <div style={{
            bottom:'5px',
            left:this.state.settings.drawerWidth * this.state.settings.showSidebar + 5 + 'px',
            position:'fixed',
            transition: 'left 0.4s ease-out'
          }}>
          <div style={{paddingBottom:4}}>
            <Fab
              style={{background:lightBlue['A400'], color:'#ffffff', zIndex:6, boxShadow:'none'}}
              aria-label="Add"
              onClick={() => {this.addFlowpoint()}}>
              <AddIcon />
            </Fab>
          </div>
          <div style={{paddingBottom:4}}>
            <Fab
              style={{background:blue['A400'], color:'#ffffff', zIndex:6, boxShadow:'none'}}
              aria-label="Copy code"
              onClick={() => {
                copy(this.state.code)
                var settings = this.state.settings;
                settings.snackshow = true;
                settings.snackmsg = 'Copied code to clipboard';
                settings.snackcolor = blue['A400'];
                this.setState({settings})
              }}>
              <FileCopyIcon />
            </Fab>
          </div>
          <div style={{paddingBottom:4}}>
            <Fab
              style={{background:indigo['A400'], color:'#ffffff', zIndex:6, boxShadow:'none'}}
              aria-label="Copy code"
              onClick={() => {
                var new_url = parseToQuery(
                  this.state.flowpoints,
                  this.state.settings.theme,
                  this.state.settings.variant,
                  this.state.settings.background,
                  this.state.count,
                  this.state.settings.darktheme,
                  this.state.settings.showName,
                  this.state.environment.library,
                  this.state.settings.showSidebar
                )
                new_url = ReplaceAll(new_url, '.', 'lll')
                postToDB(new_url, (mod_id) => {
                  new_url = this.baseUrl + '?p=' + mod_id;
                  copy(new_url)
                  var settings = this.state.settings;
                  settings.snackshow = true;
                  settings.snackmsg = 'Copied link to clipboard';
                  settings.snackcolor = indigo['A400'];
                  settings.modelUrl = new_url
                  this.setState({settings})
                })
              }}>
              <LinkIcon />
            </Fab>
          </div>
          <div>
            <Fab
              style={{background:deepPurple['A400'], color:'#ffffff', zIndex:6, boxShadow:'none'}}
              aria-label="Hide/Show"
              onClick={() => {
                var settings = this.state.settings;
                settings.showSidebar = !settings.showSidebar;
                this.setState({settings})
              }}>
              <MenuIcon />
            </Fab>
          </div>
        </div>

        <Snackbar
          autoHideDuration={3000}
          onClose={() => {
            var settings = this.state.settings;
            settings.snackshow = false
            this.setState({ settings })
          }}
          anchorOrigin={{vertical: 'bottom', horizontal: 'right'}}
          open={this.state.settings.snackshow}>
          <SnackbarContent
            message={this.state.settings.snackmsg}
            style={{backgroundColor:this.state.settings.snackcolor, color:'black'}}/>
        </Snackbar>

      </div>
    );
  }
}

export default App;
