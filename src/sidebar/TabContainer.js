import React from 'react';

import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';

import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';


// Styling
const styles = theme => ({
  root: {
    flexGrow: 0,
    overflow:'scroll',
  },
  tabsRoot: {
    borderBottom: '1px solid #1890ff',
  },
  tabsIndicator: {
    backgroundColor: '#1890ff',
  },
  tabRoot: {
    textTransform: 'initial',
    minWidth: 50,
    fontWeight: theme.typography.fontWeightRegular,
    marginRight: theme.spacing.unit * 0,
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
    '&:hover': {
      color: '#40a9ff',
      opacity: 1,
    },
    '&$tabSelected': {
      color: '#1890ff',
    },
    '&:focus': {
      color: '#40a9ff',
    },
  },
  tabSelected: {},
  typography: {
    padding: theme.spacing.unit * 0,
  },
});


// Main class
class TabContainer extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      value: props.tab !== undefined ? props.tabs.indexOf(props.tab) : 0
    }
  }

  render() {
    const { classes } = this.props;
    const { value } = this.state;
    const width = Math.ceil(this.props.width/this.props.tabs.length) + 'px'

    return (
      <div className={classes.root} style={{overflow:'hidden'}}>
        <Tabs
          value={value}
          onChange={(e, value) => {
            this.setState({ value });
            if (this.props.onSelectTab) this.props.onSelectTab(this.props.tabs[value]);
          }}
          textColor="primary"
          classes={{ root: classes.tabsRoot, indicator: classes.tabsIndicator }}
          scrollButtons={false}>
          {
            this.props.tabs.map(value => {
              return (
                <Tab
                  disableRipple
                  classes={{ root: classes.tabRoot, selected: classes.tabSelected }}
                  label={value}
                  style={{width: width}}
                />
              )
            })
          }
        </Tabs>
      </div>
    );
  }
}

// Setting proptypes
TabContainer.propTypes = {
  classes: PropTypes.object.isRequired,
};

// Exporting
export default withStyles(styles)(TabContainer);
