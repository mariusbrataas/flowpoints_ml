import React, { Component } from 'react';
import './FrontHelpers.css'

// Material
import FormControl from '@material-ui/core/FormControl';
import { Paper, InputLabel, Select, MenuItem, FormControlLabel, Switch, TextField, Tooltip, Chip } from '@material-ui/core';


export const TextFieldContainer = props => {
  return (
    <TextField
      label={props.label}
      value={props.value}
      placeholder={props.placeholder}
      type={props.type}
      rows={props.rows}
      variant={props.variant}
      margin={props.margin ? props.margin : 'tight'}
      style={props.style}
      multiline={props.multiline}
      onChange={e => {
        if (props.onChange) props.onChange(e.target.value)
      }}/>
  )
}


export const SelectContainer = props => {
  return (
    <FormControl style={props.style}>
        <InputLabel htmlFor='selectContainer'>{props.label}</InputLabel>
        <Select
          value={props.value}
          inputProps={{ name:'selectCont', id:'selectContainer' }}
          onChange={e => {
            if (props.onChange) props.onChange(e.target.value)
          }}>
          {
            props.options.map(opt => {
              return (
                <MenuItem value={opt}>{opt}</MenuItem>
              )
            })
          }
        </Select>
    </FormControl>
  )
}


export const SelectContainerTooltips = props => {
  return (
    <FormControl style={props.style}>
        <InputLabel htmlFor='selectContainer'>{props.label}</InputLabel>
        <Select
          value={props.value}
          inputProps={{ name:'selectCont', id:'selectContainer' }}
          onChange={e => {
            if (props.onChange) props.onChange(e.target.value)
          }}>
          {
            Object.keys(props.options).map(opt => {
              var chips = []
              props.options[opt].map(val => {
                chips.push(
                  <Chip label={val === 'pytorch' ? 'PT' : 'TF'} style={{fontSize:10, height:'85%', marginLeft:2, color:'white', backgroundColor:(val === 'pytorch' ? '#90caf9' : '#ffcc80')}}/>
                )
              })
              return (
                <MenuItem value={opt}>
                <div class='container'>
                  <div class='option'>{opt}</div>
                  <div class='chips'>{chips}</div>
                </div>
                </MenuItem>
              )
            })
          }
        </Select>
    </FormControl>
  )
}


export class Autosuggest extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      msg: this.props.value,
      initmsg: this.props.value,
      options: this.props.options,
      open: false,
      arrow_idx: 0
    }
    this.wrapperRef = null;
    this.handleClickOutside = this.handleClickOutside.bind(this);
  }
  componentDidMount() {
    document.addEventListener('mousedown', this.handleClickOutside);
  }
  componentWillUnmount() {
    document.removeEventListener('mousedown', this.handleClickOutside);
  }
  handleClickOutside(e) {
    if (this.wrapperRef && !this.wrapperRef.contains(e.target)) this.setState({open:false, msg:this.props.value})
  }
  render() {
    if (this.state.initmsg !== this.props.value) {
      this.setState({
        msg: this.props.value,
        initmsg: this.props.value,
        open: false,
        arrow_idx: 0
      })
    }
    var options = [];
    var opt_keys = [];
    var idx = 0;
    Object.keys(this.state.options).map(opt => {
      var chips = [];
      this.state.options[opt].map(val => {
        chips.push(
          <Chip label={val === 'pytorch' ? 'PT' : 'TF'} style={{fontSize:10, height:'85%', marginLeft:2, color:'white', backgroundColor:(val === 'pytorch' ? '#90caf9' : '#ffcc80')}}/>
        )
      })
      if (opt.toLowerCase().includes(this.state.msg.toLowerCase())) {
        options.push(
          <MenuItem
            value={opt}
            selected={this.state.arrow_idx === idx}
            onClick={e => {
              this.setState({arrow_idx:idx, msg:opt, open:false})
              if (this.props.onChange) this.props.onChange(opt)
            }}>
            <div class='container'>
              <div class='option'>{opt}</div>
              <div class='chips'>{chips}</div>
            </div>
          </MenuItem>
        )
        opt_keys.push(opt)
        idx += 1
      }
    })
    if (options.length === 0) options.push( <MenuItem disabled value='Nothing'>No available layers</MenuItem> )
    return (
      <div style={{width:'100%'}}>
        <TextField
          style={{width:'100%', paddingTop:0, marginTop:0}}
          onClick={() => {this.setState({open:!this.state.open})}}
          label="Layer type"
          value={this.state.msg}
          onChange={e => {this.setState({msg:e.target.value, arrow_idx:0, open:true})}}
          margin="normal"
          onKeyDown={e => {
            if (e.keyCode === 40) {
              this.setState({arrow_idx: Math.max(Math.min(options.length - 1, this.state.arrow_idx + 1), 0)})
            } else if (e.keyCode === 38) {
              this.setState({arrow_idx: Math.min(Math.max(0, this.state.arrow_idx - 1), options.length)})
            } else if (e.keyCode === 13) {
              if (opt_keys.length > this.state.arrow_idx) {
                this.setState({msg:opt_keys[this.state.arrow_idx], open:false})
                if (this.props.onChange) this.props.onChange(opt_keys[this.state.arrow_idx])
              }
            }
          }}
        />
        {
          this.state.open ? <div ref={node => {this.wrapperRef = node}}>
            <Paper style={{maxHeight:'40vh', overflow:'scroll'}}>
              {
                options
              }
            </Paper></div> : null
        }
      </div>
    )
  }
}


/*export const SwitchContainer = props => {
  return (
    <FormControlLabel
      style={props.style}
      control={
        <Switch
          checked={props.value}
          color='primary'
          onChange={e => {
            if (props.onChange) props.onChange(!props.value)
          }}/>
      }
      label={props.label}/>
  )
}*/


export const SwitchContainer = props => {
  return (
    <Chip
      label={props.label}
      clickable
      style={{
        marginLeft:5,
        marginTop:2,
        marginBottom:2,
        backgroundColor:(props.value ? '#64b5f6' : '#e0e0e0'),
        color:(props.value ? 'white' : '#424242')
      }}
      onClick={e => {
        if (props.onChange) props.onChange(!props.value)
      }}/>
  )
}

export const themes = [
  'red',
  'pink',
  'purple',
  'deep-purple',
  'indigo',
  'blue',
  'light-blue',
  'green',
  'light-green',
  'lime',
  'yellow',
  'amber',
  'orange',
  'deep-orange',
  'brown',
  'grey',
  'blue-grey',
  'black',
  'white'
]

export const variants = [
  'paper',
  'outlined',
  'filled'
]