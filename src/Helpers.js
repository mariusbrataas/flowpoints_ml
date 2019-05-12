import React, { Component } from 'react';

var fs = require('fs');

export function loadJSON(path) {
  return JSON.parse(fs.readFileSync(path, 'utf8'))
}

export function saveJSON(data, path) {
  fs.writeFileSync(path, JSON.stringify(data)); 
}

export function getBaseModules() {
  return loadJSON('./libraries/base_library.json')
}