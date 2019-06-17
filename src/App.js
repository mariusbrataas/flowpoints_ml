import React, { Component } from 'react';
import './App.css';

// Importing installed tools
import { Flowpoint, Flowspace } from 'flowpoints';
import copy from 'copy-to-clipboard';


// Importing local tools
import { Sidebar } from './sidebar/Sidebar.js';
import { MainButtons } from './MainButtons';
import { Parser } from './parser/Parser';
import { Library2String, String2Library } from './LibraryParser';
import { PostToDataBase, LoadDataBase } from './DataBaseHandler';
import { Encrypt, Decrypt } from './Cryptographer';
import { Snackbar, SnackbarContent } from '@material-ui/core';
import { LoadDialog, SaveDialog } from './PasswordDialog.js';
import { MainLibrary } from './MainLibrary';
import { HelpDialog } from './HelpDialog';


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


class App extends Component{
  
  constructor(props) {
    super(props);

    // Building state library
    this.state = MainLibrary()

    // Helpers
    this.diagramRef = null;

    // Binding class methods
    this.showNotification = this.showNotification.bind(this);
    this.updateCode = this.updateCode.bind(this);
    this.updateAvailableLayers = this.updateAvailableLayers.bind(this);
    this.prepOutputShapes = this.prepOutputShapes.bind(this);
    this.loadDecryptedModel = this.loadDecryptedModel.bind(this);
    this.getEmptyFlowpointContent = this.getEmptyFlowpointContent.bind(this);
    this.addFlowpoint = this.addFlowpoint.bind(this);
    this.copyCode = this.copyCode.bind(this);
    this.createLink = this.createLink.bind(this);
    this.showHideHelp = this.showHideHelp.bind(this);
    this.showHide = this.showHide.bind(this);
    this.deleteFlowpoint = this.deleteFlowpoint.bind(this);
    this.deleteSelected = this.deleteSelected.bind(this);
    this.handleClick = this.handleClick.bind(this);

  }

  
  componentDidMount() {

    // Open drawer
    var visual = this.state.visual;
    visual.drawerOpen = true;
    this.setState({visual});

    // Loading model?
    var query = window.location.href.split(this.state.settings.baseUrl)[1]
    if (query.includes('p=')) {

      query = query.substring(query.indexOf('p=') + 2, query.length)
      query = query.substring(0, 12)

      // Notify user that model is trying to load
      this.showNotification('Trying to load model...', 'info')

      // Loading database
      LoadDataBase(data => {

        // Model in data?
        if (query in data) {
          var decrypted = Decrypt(data[query], 'Hello world')
          // Encryption?
          if (decrypted) {
            this.loadDecryptedModel(decrypted, query)
          } else {
            this.showNotification('Model is encrypted')
            var visual = this.state.visual;
            var environment = this.state.environment;
            var settings = this.state.settings;
            settings.modelID = query;
            environment.encrypted_model = data[query]
            visual.show_load_dialog = true;
            this.setState({visual, environment, settings})
          }

        } else {
          this.showNotification('Could not find model in database', 'error')
        }

      })

    }

    // Updating available layers
    this.updateAvailableLayers()

  }


  showNotification(msg, color) {
    if (!((color || 'nothing').includes('#'))) {
      switch(color) {
        case 'info':
          color = '#2979ff'
          break;
        case 'error':
          color = '#dd2c00'
          break;
        case 'warning':
          color = '#ffd600'
          break;
        case 'success':
          color = '#00b843'
          break;
        default:
          color = '#37474f'
      }
    }
    var notification = this.state.notification;
    notification.queue.push({
      msg,
      color,
      key: new Date().getTime()
    })
    if (notification.show) {
      notification.show = false;
    } else {
      if (notification.queue.length) {
        notification.content = notification.queue.shift()
        notification.show = true
      }
    }
    this.setState({notification})
  }


  updateCode(cb) {
    var environment = this.state.environment;
    let tmp = Parser(this.state)
    environment.code = tmp.msg;
    environment.order = tmp.order;
    environment.dummies = tmp.dummies;
    this.setState({environment})
    this.prepOutputShapes();
    if (cb) cb(environment.code)
  }


  updateAvailableLayers() {

    var environment = this.state.environment;

    // Creating list
    var availableLayers = {Input: Object.keys(environment.libraryFetchers)};
    Object.keys(environment.baseLib).map(layer_key => {
      availableLayers[layer_key] = Object.keys(environment.baseLib[layer_key])
    })

    // Updating environment
    environment.availableLayers = availableLayers;

    // Updating state
    this.setState({environment})

  }


  prepOutputShapes(cb) {

    if (this.state.environment.library in this.state.environment.autoparams) {

      // Helpers
      const autoparams = this.state.environment.autoparams[this.state.environment.library]();
      var flowpoints = this.state.flowpoints;
      var dummies = this.state.environment.dummies;
      const order = this.state.environment.order;
      const library = this.state.environment.library;

      // Setting output-shapes of inputs
      var visited = []
      order.map(key => {
        var point = flowpoints[key]
        if (point.base_ref === 'Input') {
          point.output_shape = Object.keys(point.content.dimensions).map(dimkey => {
            return parseInt(point.content.dimensions[dimkey])
          })
          visited.push(key)
        }
      })

      // Setting all output-shapes
      order.map(key => {
        if (!visited.includes(key)) {
          var point = flowpoints[key];
          if (point.content[library]) {
            var tmp_autoparams = autoparams[point.content[library].reference];
            if (tmp_autoparams) {
              var bestInp = null
              dummies[key].inputs.map(inp_key => {
                if (visited.includes(inp_key)) bestInp = inp_key
              })
              if (flowpoints[bestInp]) {
                const prevshape = flowpoints[bestInp].output_shape.map(value => 1 * value);
                point.content[library].parameters = tmp_autoparams.autoparams(prevshape, point.content[library].parameters)
                point.output_shape = tmp_autoparams.outshape(prevshape, point.content[library].parameters).map(value => parseInt(value))
                visited.push(key)
              }
            }
          } else {
            point.output_shape = []
          }
        }
      })

      // Updating state
      this.setState({ flowpoints })

    }

  }


  loadDecryptedModel(decrypted, model_id) {

    // Parsing
    var new_state = String2Library(decrypted, this.getEmptyFlowpointContent, this.state)

    // Updating visual
    new_state.visual.show_load_dialog = false;
    new_state.visual.load_dialog_error = false;
    new_state.visual.show_save_dialog = false;

    // Ensuring getbaselib and library fetchers are added
    new_state.environment.getBaseLibrary = this.state.environment.getBaseLibrary
    new_state.environment.libraryFetchers = this.state.environment.libraryFetchers

    // Fixing model ID
    new_state.settings.modelID = model_id || this.state.settings.modelID;

    // Setting state
    this.setState({flowpoints:{}, settings:{...this.state.settings, count:0}}, () => {
      this.setState(new_state, () => {
        // Showing notification
        this.showNotification('Loaded model', '#00b24a')
  
        // Updating code and layers
        this.updateAvailableLayers()
        this.updateCode()
      })
    })

  }


  getEmptyFlowpointContent(base_ref) {

    // Input nodes are treated differently from all other nodes
    if (base_ref === 'Input') {
      return {
        n_dims: 4,
        dimensions: {
          0: 1,
          1: 1,
          2: 1,
          3: 1
        }
      }
    }

    // Not an input node? Getting base_library references
    var environment = this.state.environment;
    const base_point = environment.getBaseLibrary()[base_ref];

    // Adding contents and returning
    var content = {};
    Object.keys(base_point).map(library_key => {
      if (library_key in environment.libraryFetchers) {
        content[library_key] = {
          reference: base_point[library_key],
          parameters: environment.libraryFetchers[library_key]()[base_point[library_key]]
        }
      }
    })
    return content

  }


  addFlowpoint() {

    // Loading from state
    var flowpoints = this.state.flowpoints;
    var settings = this.state.settings;

    // Creating flowpoint
    const base_ref = settings.count === 0 ? 'Input' : 'Linear';
    var newPoint = {
      base_ref: base_ref,
      name: '',
      outputs: {},
      is_output: false,
      concat_inputs: false,
      concat_dim: 0,
      output_shape: [],
      contiguous: false,
      reshape_ndims: 0,
      reshape_dims: [],
      pos: {
        x: settings.lastPos.x,
        y: settings.lastPos.y + 90
      },
      content: this.getEmptyFlowpointContent(base_ref)
    }

    // Adding flowpoint
    flowpoints['' + settings.count] = newPoint;

    // Connecting previously selected flowpoint to this one (maybe)
    if (settings.selected) flowpoints[settings.selected].outputs['' + settings.count] = {}

    // Updating settings
    settings.selected = '' + settings.count
    settings.count += 1
    settings.lastPos = {
      x: settings.lastPos.x,
      y: settings.lastPos.y + 90
    }

    // Updating state
    this.setState({
      flowpoints,
      settings
    })

    this.updateCode()

  }


  copyCode() {
    this.updateCode(code => {
      copy(code);
      this.showNotification('Code copied to clip-board', 'info')
    })
  }


  createLink() {
    var visual = this.state.visual;

    // Showing encryption dialog
    visual.show_save_dialog = true

    // Updating state
    this.setState({visual})

  }


  showHideHelp() {
    var visual = this.state.visual;
    visual.show_help_dialog = !visual.show_help_dialog;
    this.setState({visual})
  }


  showHide() {
    
    // Loading from state
    var visual = this.state.visual;
    
    // Open/close drawer
    visual.drawerOpen = !visual.drawerOpen;

    // Updating state
    this.setState({ visual })

  }


  deleteFlowpoint(key) {

    // Loading from state
    var flowpoints = this.state.flowpoints;
    var settings = this.state.settings;

    // Removing selected?
    if (key === settings.selected) settings.selected = null;

    // Deleting flowpoint
    delete flowpoints[key];

    // Removing any connections other flowpoints have to the deleted one
    Object.keys(flowpoints).map(test_key => {
      if (key in flowpoints[test_key].outputs) {
        delete flowpoints[test_key].outputs[key]
      }
    })

    // Updating state
    this.setState({
      flowpoints,
      settings
    })

  }


  deleteSelected() {
    this.deleteFlowpoint(this.state.settings.selected);
  }


  handleClick(key, e) {

    // Loading from state
    var flowpoints = this.state.flowpoints;
    var settings = this.state.settings;

    // Handling click
    if (e.shiftKey) {
      // If shift is pressed: Create / delete connection
      if (settings.selected === null) {
        // Nothing selected: selecting current
        settings.selected = key;
      } else {
        if (settings.selected !== key) {
          // Creating connection from previously clicked to current clicked
          var pointA = flowpoints[settings.selected]
          if (key in pointA.outputs) {
            delete pointA.outputs[key]
          } else {
            pointA.outputs[key] = {}
          }
          this.updateCode()
        }
      }
    } else {
      // If not shift pressed: Select / deselect flowpoint
      settings.selected = (settings.selected === null ? key : (settings.selected == key ? null : key))
    }

    // Updating state
    this.setState({
      flowpoints,
      settings
    })

  }


  render() {
    return (
      <div style={{backgroundColor: (this.state.visual.darkTheme ? 'black' : 'white')}}>


        <Sidebar
          state={this.state}
          refresh={() => {return this.state}}
          updateFlowpoints={flowpoints => {this.setState({flowpoints}); this.updateCode()}}
          updateEnvironment={environment => {this.setState({environment}); this.updateCode()}}
          updateVisual={visual => this.setState({visual})}
          updateSettings={settings => {this.setState({settings}); this.updateCode()}}
          notification={(msg, color) => this.showNotification(msg, color)}
          getEmptyFlowpointContent={this.getEmptyFlowpointContent}
          deleteSelected={this.deleteSelected}
          updateAvailableLayers={this.updateAvailableLayers}
          diagramRef={this.diagramRef}
          prepOutputShapes={this.prepOutputShapes}/>
        

        <Flowspace
          theme={this.state.visual.theme}
          variant={this.state.visual.variant}
          background={this.state.visual.darkTheme ? 'black' : 'white'}
          selected={this.state.settings.selected}
          getDiagramRef={ref => {this.diagramRef = ref}}
          avoidCollisions
          onClick={() => {
            
            // Loading from state
            var settings = this.state.settings;
            settings.selected = null;

            // Updating state
            this.setState({settings})

          }}
          style={{
            height: '100vh',
            width: 'calc(100vw - ' + this.state.visual.drawerWidth * this.state.visual.drawerOpen + ')',
            marginLeft: this.state.visual.drawerWidth * this.state.visual.drawerOpen,
            transition: 'margin-left 0.4s ease-out'
          }}>

          {
            Object.keys(this.state.flowpoints).map(key => {
              const point = this.state.flowpoints[key];
              return (
                <Flowpoint
                  key={key}
                  outputs={point.outputs}
                  onClick={e => {this.handleClick(key, e)}}
                  startPosition={point.pos}
                  snap={{x:10, y:10}}
                  style={{
                    width:'auto',
                    height:'auto',
                    minWidth:150,
                    maxHeight: (this.state.visual.showShape && this.state.environment.library in this.state.environment.autoparams) ? 150 : 50
                  }}
                  onDrag={pos => {
                    var flowpoints = this.state.flowpoints;
                    var settings = this.state.settings;
                    flowpoints[key].pos = pos;
                    settings.lastPos = pos;
                    this.setState({flowpoints, settings})
                  }}>
                  <div style={{height:'auto', paddingLeft:4, paddingRight:4}}>
                      <div style={{display:'table', width:'100%', height:'50px'}}>
                        <div style={{display:'table-cell', verticalAlign:'middle', textAlign:'center'}}>
                          {
                            this.state.visual.showName ? (point.name !== '' ? point.name : 'p_' + key) : point.base_ref
                          }
                        </div>
                      </div>
                      {
                        (this.state.visual.showShape && this.state.environment.library in this.state.environment.autoparams) ? shapeBox(point.output_shape) : null
                      }
                  </div>
                </Flowpoint>
              )
            })
          }

        </Flowspace>


        <MainButtons
          state={this.state}
          addFlowpoint={this.addFlowpoint}
          copyCode={this.copyCode}
          createLink={this.createLink}
          showHide={this.showHide}
          showHideHelp={this.showHideHelp}/>
        

        <Snackbar
          autoHideDuration={4000}
          onClose={() => {
            var notification = this.state.notification;
            notification.show = false;
            this.setState({notification})
          }}
          onExited={() => {
            var notification = this.state.notification;
            if (notification.queue.length > 0) {
              notification.content = notification.queue.shift();
              notification.show = true
            }
            this.setState({notification})
          }}
          anchorOrigin={{vertical:'top', horizontal:'right'}}
          open={this.state.notification.show}>
          <SnackbarContent
            message={this.state.notification.content.msg}
            style={{backgroundColor:this.state.notification.content.color, boxShadow:'none'}}/>
        </Snackbar>


        <HelpDialog
          open={this.state.visual.show_help_dialog}
          onClose={() => {
            var visual = this.state.visual;
            visual.show_help_dialog = false;
            this.setState({visual})
          }}/>

        
        <LoadDialog
          error={this.state.visual.load_dialog_error}
          open={this.state.visual.show_load_dialog}
          onClose={() => {
            var visual = this.state.visual;
            visual.show_load_dialog = false;
            this.setState({visual, settings:{...this.state.settings, modelID:null}})
          }}
          onSubmit={pswd => {
            var environment = this.state.environment;
            var decrypted = Decrypt(environment.encrypted_model, pswd)
            if (decrypted) {
              this.loadDecryptedModel(decrypted)
            } else {
              var visual = this.state.visual;
              visual.load_dialog_error = true;
              this.setState({visual})
              this.showNotification('Wrong password', 'error')
            }
          }}/>


        <SaveDialog
          error={this.state.visual.save_dialog_error}
          open={this.state.visual.show_save_dialog}
          onClose={() => {
            var visual = this.state.visual;
            visual.show_save_dialog = false;
            this.setState({visual})
          }}
          onSubmit={pswd => {
            var visual = this.state.visual;
            visual.show_save_dialog = false;
            this.setState({visual})
            PostToDataBase(Encrypt(Library2String(this.state), (pswd === '' ? 'Hello world' : pswd)), model_id => {

              // Loading from state
              var settings = this.state.settings;
              var newUrl = settings.baseUrl + '?p=' + model_id;

              // Setting model id
              settings.modelID = model_id;

              // Updating state
              this.setState({settings})

              // Updating code
              this.updateCode()

              // Changing current url
              window.history.pushState({}, model_id, newUrl);

              // Copy link to clip-board and display notification
              copy(newUrl)
              this.showNotification('Model saved and link copied to clip-board')

            })
          }}/>
      

      </div>
    )
  }
}

export default App;
