//var GoogleSpreadsheet = require('google-spreadsheet');
//var request = require('request');
import axios from 'axios'

function ReplaceAll(str, search, replacement) {
  var msg = ''
  str.split(search).map(sub => {
    msg += sub + replacement
  })
  return msg.substring(0, msg.length - replacement.length)
}

function num2string(num) {
  return num.toString(36)
}
//AIzaSyC2xSmTdMOwyfYJNWgRI0AZolpKgPft8a0
function string2num(str) {
  return parseInt(str, 36)
}
// 60 12 99 24 60
function getId(l) {
  l = Math.min(20, Math.max(10, l || 15))
  var d = new Date()
  var n = ''
  n += '' + d.getSeconds()
  n += '' + d.getMonth()
  n += '' + d.getYear().toString().substring(1)
  n += '' + d.getHours()
  n += '' + d.getMinutes()
  var msg = num2string(parseInt(n))
  const lib = 'abcdefghijklmnopqrstuvwxyz0123456789'
  Array.from(Array(l - msg.length).keys()).map(idx => {
    msg = lib[Math.floor(Math.random() * lib.length)] + msg;
  })
  return msg
}

export function postToDB(content, cb) {
  const mod_id = getId();
  var url = 'https://docs.google.com/forms/d/e/1FAIpQLSfTgXs4tTJ5drC44SVHCJaJoIwogOwK6WBrGT2-s85Ndaqz_g/formResponse?usp=pp_url'
  url += '&entry.865206380=' + mod_id;
  url += '&entry.854684610=' + content;
  url += '&submit=Submit'
  axios.get(url).then(res => {})
  cb(mod_id)
}

export function getDB(cb) {
  var res;
  axios.get('https://spreadsheets.google.com/feeds/list/11Z7JHNLrE0ODKnGwU0NOblcV-OxRcbugr8ujNZDzTRA/od6/public/basic?alt=json').then(res => {
    var lib = {}
    res.data.feed.entry.map(entry => {
      lib[entry.title['$t']] = entry.content['$t'].split('content: ')[1]
    })
    cb(lib)
  })
}
