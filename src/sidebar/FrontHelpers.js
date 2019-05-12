import React, { Component } from 'react';
import './FrontHelpers.css'

// Material
import FormControl from '@material-ui/core/FormControl';
import { InputLabel, Select, MenuItem, FormControlLabel, Switch, TextField, Tooltip, Chip } from '@material-ui/core';


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