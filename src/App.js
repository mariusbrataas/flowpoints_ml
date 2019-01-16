import React, { Component } from 'react';
import './App.css';

import Snackbar from '@material-ui/core/Snackbar';

// Importing local tools
import { Codepaper } from './flowparser/Codepaper.js'
import { AppBottom, gen_url, parse_url } from './helpers/AppBottom.js'
import { DrawConnections } from './helpers/DrawConnections.js'
import { DrawPoints } from './flowpoint/DrawPoints.js'
import { TopBar } from './helpers/TopBar.js'
import { getOrdered, getInputs } from './flowparser/FlowOrder'


class App extends Component {
  constructor(props) {
    super(props);
    this.bezierOffset = 90
    this.lastPosX = 0
    this.lastPosY = 0
    this.maxX = 0
    this.maxY = 0
    this.state = {
      flowpoints: {},
      settings: {
        count: 0,
        zoom: 1.0,
        currentInput: null,
        currentOutput: null,
        showPaper: false,
        snapX: 10,
        snapY: 10,
        showSnackbar: false,
        snackbarMsg: 'Wubbalubbadubdub',
        url: 'https://mariusbrataas.github.io/flowpoints_ml'
      }
    };
    this.updateLastPos = this.updateLastPos.bind(this)
    this.updateFrameSize = this.updateFrameSize.bind(this)
    this.autoFrameSize = this.autoFrameSize.bind(this)
    this.setOutputShapes = this.setOutputShapes.bind(this)
  }
  updateLastPos(x, y) {
    this.lastPosX = x
    this.lastPosY = y
  }
  updateFrameSize(x, y) {
    this.maxX = x
    this.maxY = y
  }
  autoFrameSize() {
    const flowpoints = this.state.flowpoints
    const mult = this.state.settings.showPaper ? 4 : 2
    var maxX = 10
    var maxY = 10
    Object.keys(flowpoints).map((key) => {
      const point = flowpoints[key]
      maxX = Math.max(maxX, point.x + mult * point.width)
      maxY = Math.max(maxY, point.y + 2 * point.height)
    })
    this.updateFrameSize(maxX, maxY)
  }
  componentWillMount() {
    const addr = window.location.href
    if (addr.includes('load?')) {
      try {
        var flowpoints = parse_url(addr.split('load?')[1])
        var settings = this.state.settings
        var max_idx = 0
        Object.keys(flowpoints).map(key => {
          max_idx = Math.max(max_idx, parseInt(key) + 1)
        })
        settings.count = max_idx
        this.setState({flowpoints, settings})
      } catch (error) {
        var settings = this.state.settings
        settings.snackbarMsg = 'Failed to load model.'
        settings.showSnackbar = true
        this.setState({settings})
      }
    }
  }
  setOutputShapes(inps, order) {
    order.map((key, index) => {
      var point = this.state.flowpoints[key]
      var params = null
      if (!point.flowtype.includes('input')) {
        if (point.flowtype in point.layertypes) {
          params = point.layertypes[point.flowtype]
        } else {
          params = point.activationtypes[point.flowtype]
        }
        var bestInp = null
        point.inputs.map(inpkey => {
          if (order.indexOf(key) > order.indexOf(inpkey)) {
            bestInp = inpkey
          }
        })
        if (bestInp != null) {
          params.params = params.autoparams(this.state.flowpoints[bestInp].output_shape, params.params)
          this.state.flowpoints[key].output_shape = params.outshape(this.state.flowpoints[bestInp].output_shape, params.params)
        }
      }
    })
  }
  render() {
    this.autoFrameSize()
    var order = []
    var inps = []
    if (Object.keys(this.state.flowpoints).length != 0) {
      inps = getInputs(this.state.flowpoints)
      order = getOrdered(this.state.flowpoints, inps)
      // Updating output shapes
      this.setOutputShapes(inps, order)
    }
    if (order.length > 0) {
      //window.history.replaceState({}, null, this.state.settings.url + '/load?' + gen_url(this.state.flowpoints, order))
      window.history.replaceState({}, null, this.state.settings.url + '/?p=' + 'load?' + gen_url(this.state.flowpoints, order))
    } else {
      window.history.replaceState({}, null, this.state.settings.url + '/')
    }
    return (
      <div>
        <div ref={this.mainRef} style={{width:'100vw', height:'90vh', paddingTop:'10vh', overflow:'scroll'}}>
          <Codepaper
            flowpoints={this.state.flowpoints}
            settings={this.state.settings}/>
          <div style={{transform:'scale(' + this.state.settings.zoom + ')', 'transform-origin':'top left', 'transition':'.1s ease-in-out'}}>
            <div style={{width:this.maxX+50, height:this.maxY+50, position:'relative'}}>
              <DrawConnections flowpoints={this.state.flowpoints}/>
              <DrawPoints
                flowpoints={this.state.flowpoints}
                updateLastPos={this.updateLastPos}
                lastPosX={this.lastPosX}
                lastPosY={this.lastPosY}
                state={this.state}
                refresh={() => {return this.state}}
                updateFlowpoints={(flowpoints) => {this.setState({flowpoints})}}
                updateSettings={(settings) => {this.setState({settings})}}
                updateView={(flowpoints, settings) => {this.setState({flowpoints, settings})}}/>
            </div>
          </div>
          <div style={{top:'0px', position:'fixed', zIndex:5}}>
            <TopBar/>
          </div>
          <AppBottom
            updateLastPos={this.updateLastPos}
            lastPosX={this.lastPosX}
            lastPosY={this.lastPosY}
            state={this.state}
            order={order}
            refresh={() => {return this.state}}
            updateFlowpoints={(flowpoints) => {this.setState({flowpoints})}}
            updateSettings={(settings) => {this.setState({settings})}}
            updateView={(flowpoints, settings) => {this.setState({flowpoints, settings})}}/>
          <Snackbar
            anchorOrigin={{vertical: 'bottom', horizontal: 'left'}}
            open={this.state.settings.showSnackbar}
            autoHideDuration={3000}
            onClose={() => {
              var settings = this.state.settings
              settings.showSnackbar = false
              this.setState({settings})
            }}
            message={this.state.settings.snackbarMsg}
            style={{width:'auto'}} />
        </div>
      </div>
    )
  }
}


export default App;
