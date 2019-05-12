import React from 'react';
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import TextField from '@material-ui/core/TextField';
import Typography from '@material-ui/core/Typography';
import Collapse from '@material-ui/core/Collapse';


class WhatIsThis extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      open: false
    }
  }
  render() {
    return (
      <div style={{padding:0}}>
        <Button color='primary' style={{textTransform:'none', padding:'1', marginLeft:-7}} onClick={() => {this.setState({open:!this.state.open}); this.props.onClick()}}>
          What is this?
        </Button>
        <div style={{marginLeft:20, marginTop:10}} onClick={() => {this.setState({open:false}); this.props.onClick()}}>
          <Collapse in={this.state.open}>
            <Typography color='inherit' paragraph>Model protection, using AES encryption.</Typography>
            <Typography color='inherit' paragraph>
              When a model is saved, it's content is converted to a string and stored in a
              publicly view-able google sheet.<br/>
              Altough anyone can see the model in this google sheet, encryption will make it
              impossible for anyone without the password to read the model.
            </Typography>
            <Typography color='inherit' paragraph>
              If you choose to encrypt your model, your own device will perform the
              encryption, and then send the encrypted model to the google sheet.<br/><br/>

              When you try to load an encrypted model, the encrypted string will be downloaded
              from the google sheet, and then decrypted by your device.<br/><br/>
              This ensures point-to-point encryption, in which no un-encrypted data is ever
              transmitted over the internet.
            </Typography>
            <Typography color='inherit' style={{fontWeight:'bold'}} paragraph>
              NB!<br/>
              The password is NEVER STORED! If you forget your password, the contents
              of your model cannot be recovered.
            </Typography>
          </Collapse>
        </div>
        <br/>
      </div>
    )
  }
}


export class LoadDialog extends React.Component {

  constructor(props) {
    super(props)
    this.state = {
      open: false,
      pswd: ''
    };
    this.fieldRef = null;
  }

  render() {
    return (
      <Dialog
        open={this.props.open}
        onClose={this.props.onClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description">
        <DialogTitle id="alert-dialog-title">Encrypted model</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Please type your password in order to decrypt the model<br/>
            <WhatIsThis onClick={() => {if (this.fieldRef) {this.fieldRef.focus()}}}/>
            <form 
              autoComplete='off'
              style={{marginTop:0, paddingTop:0}}
              onSubmit={(e) => {
                e.preventDefault();
                if (this.props.onSubmit) {this.props.onSubmit(this.state.pswd)}
              }}>
              <TextField
                id="pswdfield"
                label={this.props.error ? 'Wrong password' : 'Password'}
                error={this.props.error}
                value={this.state.pswd}
                onChange={(e) => {
                  this.setState({pswd:e.target.value})
                }}
                type="text"
                InputLabelProps={{
                  shrink: true,
                }}
                style={{width:'100%', marginTop:0, paddingTop:0}}
                margin="normal"
                inputRef={(input) => {if (input) {input.focus(); this.fieldRef = input}}}/>
            </form>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              if (this.state.pswd === '') {
                this.props.onClose()
              } else {
                this.props.onSubmit(this.state.pswd)
              }
            }}
            color="primary">
            {
              (this.state.pswd === '') ? (this.props.continueMsg ? this.props.continueMsg : 'CONTINUE WITHOUT LOADING MODEL') : 'DECRYPT MODEL'
            }
          </Button>
        </DialogActions>
      </Dialog>
    );
  }
}


export class SaveDialog extends React.Component {

  constructor(props) {
    super(props)
    this.state = {
      open: true,
      pswd: ''
    };
    this.fieldRef = null;
  }

  render() {
    return (
      <Dialog
        open={this.props.open}
        onClose={this.props.onClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description">
        <DialogTitle id="alert-dialog-title">Model encryption (optional)</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Type a password to encrypt your model, or leave the
            password field empty in order to create a public link<br/>
            <WhatIsThis onClick={() => {if (this.fieldRef) {this.fieldRef.focus()}}}/>
            <form 
              autoComplete='off'
              style={{marginTop:0, paddingTop:0}}
              onSubmit={(e) => {
                e.preventDefault();
                if (this.props.onSubmit) this.props.onSubmit(this.state.pswd)
              }}>
              <TextField
                id="pswdfield"
                label={this.props.error ? 'Wrong password' : 'Password'}
                error={this.props.error}
                value={this.state.pswd}
                onChange={(e) => {
                  this.setState({pswd:e.target.value})
                }}
                type="text"
                InputLabelProps={{
                  shrink: true,
                }}
                style={{width:'100%', marginTop:0, paddingTop:0}}
                margin="normal"
                inputRef={(input) => {if (input) {input.focus(); this.fieldRef = input}}}/>
            </form>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              this.props.onSubmit(this.state.pswd)
            }}
            color="primary">
            {
              (this.state.pswd === '') ? (this.props.continueMsg ? this.props.continueMsg : 'Continue without password') : 'Continue'
            }
          </Button>
        </DialogActions>
      </Dialog>
    );
  }
}