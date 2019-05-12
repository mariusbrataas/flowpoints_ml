import React, { Component } from 'react';

// Styles
import './Sidebar.css';

// Material
import { MuiThemeProvider, createMuiTheme } from '@material-ui/core/styles';
import Drawer from '@material-ui/core/Drawer';

// Local tools
import { SidebarHead } from './SidebarHead.js';
import { SidebarContents } from './SidebarContents.js';
import TabContainer from './TabContainer.js';


// Themes
const darkMuiTheme = createMuiTheme({
  palette: {
    type: 'dark'
  },
  typography: { useNextVariants: true }
});

const lightMuiTheme = createMuiTheme({
  palette: {
    type: 'light'
  },
  typography: { useNextVariants: true }
});


// Sidebar
export const Sidebar = props => {
  var state = props.state;
  var visual = state.visual;
  var settings = state.settings;
  return (
    <MuiThemeProvider theme={visual.darkTheme ? darkMuiTheme : lightMuiTheme}>

      <Drawer variant='persistent' anchor='left' open={visual.drawerOpen}>
        <div
          class='fullSidebar'
          style={{
            maxWidth: visual.drawerWidth,
            color: (visual.darkTheme ? 'white' : 'black'),
            width: visual.drawerWidth
          }}>

          <SidebarHead/>

          <div>
            <TabContainer
              tabs={['Misc', 'Code', 'Flowpoint']}
              tab={settings.tab}
              width={360}
              onSelectTab={tab => {

                // Loading state
                var state = props.refresh();
                var settings = state.settings;
                var visual = state.visual;

                // Changing tab
                settings.tab = tab;

                // Wider drawer if displaying code
                if (tab === 'Code') {
                  visual.drawerWidth = 600;
                } else {
                  visual.drawerWidth = 360;
                }

                // Updating state
                props.updateSettings(settings);
                props.updateVisual(visual);

              }}/>
          </div>
          
          <SidebarContents
            state={state}
            refresh={props.refresh}
            updateFlowpoints={props.updateFlowpoints}
            updateEnvironment={props.updateEnvironment}
            updateVisual={props.updateVisual}
            updateSettings={props.updateSettings}
            notification={props.notification}
            getEmptyFlowpointContent={props.getEmptyFlowpointContent}
            deleteSelected={props.deleteSelected}
            updateAvailableLayers={props.updateAvailableLayers}
            diagramRef={props.diagramRef}
            prepOutputShapes={props.prepOutputShapes}/>

        </div>
      </Drawer>

    </MuiThemeProvider>
  )
}