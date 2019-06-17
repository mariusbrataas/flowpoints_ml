import React from 'react';
import { Typography, Dialog, DialogTitle, DialogContentText, DialogContent, ExpansionPanel, ExpansionPanelSummary, Stepper, StepContent, StepLabel, Step, Link } from "@material-ui/core";


export class HelpDialog extends React.Component{
  constructor(props) {
    super(props);
    this.state = { step:0 }
    this.goTo = this.goTo.bind(this);
  }

  goTo(idx) {
    this.setState({ step:idx })
  }

  render() {
    return (
      <Dialog
        open={this.props.open}
        onClose={this.props.onClose}
        fullWidth
        maxWidth='sm'
        style={{maxHeight:'90vh', height:'auto'}}>

        <DialogTitle>Getting started</DialogTitle>

        <div style={{overflow:'scroll', width:'100%'}}>

          <DialogContent>
            <DialogContentText>
              Welcome to Flowpoints ML!<br/>
              Here you can design deep learning models graphically.<br/><br/>
              The following few sections will help you get going :)<br/>
              You can also check out the <Link href='https://github.com/mariusbrataas/flowpoints_ml#readme' target='_blank'>documentation</Link> for more information.<br/><br/>
              Found an bug? Please open a new <Link href="https://github.com/mariusbrataas/flowpoints_ml/issues" target='_blank'>issue</Link>, and feel free to fix it!
            </DialogContentText>
          </DialogContent>

          <Stepper activeStep={this.state.step} orientation='vertical'>

            <Step key={0} completed={false}>
              <StepLabel onClick={() => {this.goTo(0)}}>The main parts of the screen</StepLabel>
              <StepContent>
                <Typography>
                  Before we get started you should know the main parts of this tool.<br/><br/>
                  
                  The little stack of colored buttons:<br/>
                  You'll use these for creating new flowpoints, copy the entire code of your model,
                  create shareable links to your model, show/hide the panel on the left of your screen,
                  and to view this guide.<br/><br/>
                  
                  The "Misc" tab in the side panel:<br/>
                  Here you can change what library you want to utilize to create your model,
                  write some notes about your work, and change the appearance of things.
                  
                  The "Code" tab in the side panel:<br/>
                  When you create a model, it's corresponding code will be displayed here.<br/><br/>
                  
                  The "Flowpoint" tab:<br/>
                  All parameters of the selected flowpoint can be viewed and changed in this tab.
                </Typography>
              </StepContent>
            </Step>

            <Step key={1} completed={false}>
              <StepLabel onClick={() => {this.goTo(1)}}>Creating a new flowpoint</StepLabel>
              <StepContent>
                <Typography>
                  Click the + button in the button stack. This should create a new
                  flowpoint on your screen and move focus to this one automatically.
                </Typography>
              </StepContent>
            </Step>

            <Step key={2} completed={false}>
              <StepLabel onClick={() => {this.goTo(2)}}>Connecting flowpoints</StepLabel>
              <StepContent>
                <Typography>
                  First create two flowpoints.<br/>
                  Did they create a connection automatically? No worries, that is supposed
                  to happen whenever you create a flowpoint while another flowpoint is selected.<br/><br/>
                  To create a new connection manually, start by selecting the flowpoint
                  you want to get the output from. Next, hold shift while clicking the flowpoint
                  that should receive the output. Poof! They're connected!<br/><br/>
                  Disconnecting flowpoints is just as simple. Select the flowpoint that supplies the output,
                  hold shift, and click the flowpoint that's receiving the input.
                </Typography>
              </StepContent>
            </Step>

            <Step key={3} completed={false}>
              <StepLabel onClick={() => {this.goTo(3)}}>Direction of connections</StepLabel>
              <StepContent>
                <Typography>
                  All connections between flowpoints signify what direction data is moving.<br/><br/>
                  When connecting two flowpoints, the first flowpoint you select will be used as
                  the "sender" in that connection, while the second flowpoint will be the "receiver".<br/><br/>
                  The color-gradient of connections help identify which is which.
                </Typography>
              </StepContent>
            </Step>

            <Step key={4} completed={false}>
              <StepLabel onClick={() => {this.goTo(4)}}>Changing a flowpoint's parameters</StepLabel>
              <StepContent>
                <Typography>
                  If you head over to the "Flowpoint" tab you should see a bunch of fields
                  (click on a flowpoint if you can't see any such fields).<br/><br/>
                  To change the value of a field, click it, and try typing something.
                  Note that some fields will only accept numbers.
                </Typography>
              </StepContent>
            </Step>

            <Step key={5} completed={false}>
              <StepLabel onClick={() => {this.goTo(5)}}>Layer types</StepLabel>
              <StepContent>
                <Typography>
                  The field at the very top in the "Flowpoint" tab is where you assign
                  a layer type to your flowpoint.<br/><br/>
                  When you click this field, a long list will pop up. These are all
                  the layers available. Small, colored badges indicate what libraries the
                  layer is available in, i.e. TF for TensorFlow, or PT for PyTorch.<br/><br/>
                  If you start typing the name of a layer, the list will try to suggest a smaller selection of layers.<br/><br/>
                  Ideally, it should be possible to create models utilizing layers that are available
                  in either library, and quickly switch between them by just changing the "Library" field
                  in the "Misc" tab.<br/>
                  This won't always be the case, but entertain this idea, paramaters from all libraries
                  will be stored whenever you create a link to your model.
                </Typography>
              </StepContent>
            </Step>

            <Step key={6} completed={false}>
              <StepLabel onClick={() => {this.goTo(6)}}>Changing appearance</StepLabel>
              <StepContent>
                <Typography>
                  Head over to the "Misc" tab. You'll see the fields "Theme" and "Variant",
                  and a couple of buttons just beneath.<br/><br/>
                  These have no practical application what-so-ever, but you'll be able to
                  create deep learning models with style! I recommend trying the theme "orange"
                  with the variant "paper".<br/><br/>
                  Play arround till you find something you like. When you share a model, any
                  changes to it's appearance will be included.
                </Typography>
              </StepContent>
            </Step>

            <Step key={7} completed={false}>
              <StepLabel onClick={() => {this.goTo(7)}}>Sharing</StepLabel>
              <StepContent>
                <Typography>
                  Ready to show off your work?<br/><br/>
                  Click the button in the button stack showing a link. This should open a new box on your
                  screen, asking wether you'd like to encrypt your model.<br/><br/>
                  Adding encryption will ensure that no one without the correct password can open your model.<br/><br/>
                  If you want to create a public link, just leave the text field empty and click continue.<br/><br/>
                  The link will be copied to your clip-board, and the current URL should at the top of your
                  browser will change in order to match the link.<br/><br/>
                  If you chose to add a password, this password will be requested the next time you open your model.<br/><br/>
                  Note that if you forget your password, the data of your model can be considered lost.
                  Your password is not stored anywhere, and there exists no other keys that can decrypt your data.
                </Typography>
              </StepContent>
            </Step>

            <Step key={8} completed={false}>
              <StepLabel onClick={() => {this.goTo(8)}}>Examples</StepLabel>
              <StepContent>
                <Typography>
                  <Link href="https://mariusbrataas.github.io/flowpoints_ml/?p=KlHpdLzP3SDx" target="_blank">TensorFlow CNN used for the CIFAR10 example</Link><br/>
                  <Link href="https://mariusbrataas.github.io/flowpoints_ml/?p=9fehu18ra4ty" target="_blank">PyTorch CNN used for the CIFAR10 example</Link>
                </Typography>
              </StepContent>
            </Step>

          </Stepper>
        </div>
      </Dialog>
    )
  }
}