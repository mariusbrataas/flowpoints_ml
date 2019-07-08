import React, { Component } from 'react';

// Local tools
import { SelectContainer, themes, variants, SwitchContainer, TextFieldContainer } from './FrontHelpers.js';
import { Button } from '@material-ui/core';

// Other imports

const htmlToImage = require('html-to-image');



const NotesSection = props => {
  return (
    <div style={{paddingTop:30}}>

      <TextFieldContainer
        label='Model notes'
        multiline={true}
        value={props.state.environment.notes}
        style={{width:'100%'}}
        onChange={val => {

          // Loading from state
          var environment = props.refresh().environment;

          // Changing notes
          environment.notes = val;

          // Updating state
          props.updateEnvironment(environment)

        }}/>

    </div>
  )
}


const EnvironmentSection = props => {
  var environment = props.state.environment;
  return (
    <div>

      <h3 style={{marginTop:0}}>Model name</h3>

      <TextFieldContainer
        label='Model name'
        multiline={true}
        value={props.state.environment.modelname}
        style={{width:'100%'}}
        onChange={val => {

          // Loading from state
          var environment = props.refresh().environment;

          // Changing notes
          environment.modelname = val;

          // Updating state
          props.updateEnvironment(environment)

        }}/>

      <h3 style={{marginTop:30}}>Environment</h3>

      <SelectContainer
        label='Library'
        value={environment.library}
        options={Object.keys(environment.libraryFetchers)}
        onChange={val => {
          
          // Loading from state
          var environment = props.refresh().environment;
          environment.library = val;

          // Updating state
          props.updateEnvironment(environment)

          // Updating available layers
          props.updateAvailableLayers()

        }}/>

      {
        environment.library === 'pytorch' ? <div style={{paddingTop:20}}>
            <SwitchContainer
              label='batch first'
              value={environment.batch_first}
              style={{paddingTop:0}}
              onChange={val => {

                // Loading from state
                var environment = props.refresh().environment;
                environment.batch_first = val;

                // Updating state
                props.updateEnvironment(environment)

            }}/>
            <SwitchContainer
              label='include training function'
              value={environment.include_training}
              style={{paddingTop:0}}
              onChange={val => {

                // Loading from state
                var environment = props.refresh().environment;
                environment.include_training = val;

                // Updating state
                props.updateEnvironment(environment)

              }}
            />
            <SwitchContainer
              label='include save and load'
              value={environment.include_saveload}
              style={{paddingTop:0}}
              onChange={val => {

                // Loading from state
                var environment = props.refresh().environment;
                environment.include_saveload = val;

                // Updating state
                props.updateEnvironment(environment)

              }}
            />
            <SwitchContainer
              label='include predict'
              value={environment.include_predict}
              style={{paddingTop:0}}
              onChange={val => {

                // Loading from state
                var environment = props.refresh().environment;
                environment.include_predict = val;

                // Updating state
                props.updateEnvironment(environment)

              }}
            />
          </div> : null
      }

    </div>
  )
}


const CustomizeViewSection = props => {
  var visual = props.state.visual;
  var environment = props.state.environment;
  return (
    <div style={{paddingTop:20, width:'100%'}}>
      
      <h3>Customize view</h3>

      <div>

        <SelectContainer
          label='Theme'
          value={visual.theme}
          options={themes}
          style={{width:'48%', paddingRight:10, paddingBottom:10}}
          onChange={val => {

            // Loading from state
            var visual = props.refresh().visual;
            visual.theme = val;

            // Updating state
            props.updateVisual(visual)

          }}/>
        
        <SelectContainer
          label='Variant'
          value={visual.variant}
          options={variants}
          style={{width:'48%', paddingBottom:10}}
          onChange={val => {

            // Loading from state
            var visual = props.refresh().visual;
            visual.variant = val;

            // Updating state
            props.updateVisual(visual)

          }}/>

      </div>


      <div style={{paddingTop:20}}>
        {
          [
            {ref:'darkTheme', label:'Dark theme'},
            {ref:'showName', label:'Show names'}
          ].map(tmp => {
            return (
              <SwitchContainer
                label={tmp.label}
                value={visual[tmp.ref]}
                style={{paddingTop:0}}
                onChange={val => {

                  // Loading from state
                  var visual = props.refresh().visual;
                  visual[tmp.ref] = val;

                  // Updating state
                  props.updateVisual(visual)

                }}/>
            )
          })
        }
        {
          environment.library in environment.autoparams ? <SwitchContainer
            label='Show shapes'
            value={visual.showShape}
            style={{paddingTop:0}}
            onChange={val => {
              var visual = props.refresh().visual;
              visual.showShape = val;
              props.updateVisual(visual);
              if (val) props.prepOutputShapes();
            }}/> : null
        }
      </div>

      {
        props.diagramRef ? <div style={{paddingTop:20}}>
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
            </Button></div> : null
      }

    </div>
  )
}



export const MiscTab = props => {
  return (
    <div style={{padding:15}}>

      <EnvironmentSection
        state={props.state}
        refresh={props.refresh}
        updateEnvironment={props.updateEnvironment}
        updateAvailableLayers={props.updateAvailableLayers}/>
      
      <CustomizeViewSection
        state={props.state}
        refresh={props.refresh}
        updateVisual={props.updateVisual}
        diagramRef={props.diagramRef}
        prepOutputShapes={props.prepOutputShapes}/>
      
      <NotesSection
        state={props.state}
        refresh={props.refresh}
        updateEnvironment={props.updateEnvironment}/>
      
    </div>
  )
}