import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import { Snackbar, SnackbarContent } from '@material-ui/core/Snackbar';
import IconButton from '@material-ui/core/IconButton';
import CloseIcon from '@material-ui/icons/Close';

const styles = theme => ({
  close: {
    padding: theme.spacing.unit / 2,
  },
});

class NotificationContainer extends React.Component {
  queue = [];

  state = {
    open: false,
    messageInfo: {},
  };

  handleClick = message => () => {
    this.queue.push({
      message,
      key: new Date().getTime(),
    });

    if (this.state.open) {
      // immediately begin dismissing current message
      // to start showing new one
      this.setState({ open: false });
    } else {
      this.processQueue();
    }
  };

  processQueue = () => {
    if (this.queue.length > 0) {
      this.setState({
        messageInfo: this.queue.shift(),
        open: true,
      });
    }
  };

  handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    this.setState({ open: false });
  };

  handleExited = () => {
    this.processQueue();
  };

  render() {
    const { classes } = this.props;
    const { messageInfo } = this.state;

    return (
      <Snackbar
        key={messageInfo.key}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        open={this.props.open}
        autoHideDuration={4000}
        onClose={this.props.onClose}
        onExited={this.handleExited}
        ContentProps={{
          'aria-describedby': 'message-id',
        }}
        message={<span id="message-id">{this.props.message}</span>}>
        <SnackbarContent
            message={this.props.message}
            style={{backgroundColor:this.props.color, boxShadow:'none'}}/>
      </Snackbar>
    );
  }
}

NotificationContainer.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(NotificationContainer);