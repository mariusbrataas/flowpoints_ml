import React, { Component } from 'react';

// Importing local tools
import { MiscTab } from './MiscTab.js';
import { CodeTab } from './CodeTab.js';
import { FlowpointTab } from './FlowpointTab.js';


// Styles
import './SidebarContents.css';

export const SidebarContents = props => {
  var state = props.state;
  var visual = state.visual;
  var settings = state.settings;
  return (
    <div
      class='sidebarContents'
      style={{
        backgroundColor: (visual.darkTheme ? '#1b1918' : null)
      }}>

      {
        settings.tab === 'Misc' ? <MiscTab
          state={state}
          refresh={props.refresh}
          updateFlowpoints={props.updateFlowpoints}
          updateEnvironment={props.updateEnvironment}
          updateVisual={props.updateVisual}
          updateSettings={props.updateSettings}
          notification={props.notification}
          updateAvailableLayers={props.updateAvailableLayers}
          diagramRef={props.diagramRef}
          prepOutputShapes={props.prepOutputShapes}/> : null
      }
      {
        settings.tab === 'Code' ? <CodeTab
          state={state}
          refresh={props.refresh}
          updateFlowpoints={props.updateFlowpoints}
          updateEnvironment={props.updateEnvironment}
          updateVisual={props.updateVisual}
          updateSettings={props.updateSettings}
          notification={props.notification}/> : null
      }
      {
        settings.tab === 'Flowpoint' ? <FlowpointTab
          state={state}
          refresh={props.refresh}
          updateFlowpoints={props.updateFlowpoints}
          updateEnvironment={props.updateEnvironment}
          updateVisual={props.updateVisual}
          updateSettings={props.updateSettings}
          notification={props.notification}
          getEmptyFlowpointContent={props.getEmptyFlowpointContent}
          deleteSelected={props.deleteSelected}/> : null
      }

    </div>
  )
}