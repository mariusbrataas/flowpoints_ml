/*
DBhandler.js

Writing to and reading from a public google sheet.
Rows are added to the sheet using forms. This is a hackish solution that
doesn't allow changing of existing entries, but it serves it's purpose well
for this project.
*/

// Imports
import axios from 'axios'

// Helper to replace all occurences of substring in string
function ReplaceAll(str, search, replacement) {
  var msg = ''
  str.split(search).map(sub => {
    msg += sub + replacement
  })
  return msg.substring(0, msg.length - replacement.length)
}

// Converting a number to a 36 numeral string representation
function num2string(num) {
  return num.toString(36)
}

// Converting back to number
function string2num(str) {
  return parseInt(str, 36)
}

// Generating id.
// IDs are kept unique (most likely) by making parts of the ID rely entirely on
// the current date and time, and having the rest generated randomly.
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

// Using google forms to add a new row to the sheet
export function postToDB(content, cb) {
  const mod_id = getId();
  var url = 'https://docs.google.com/forms/d/e/1FAIpQLSfTgXs4tTJ5drC44SVHCJaJoIwogOwK6WBrGT2-s85Ndaqz_g/formResponse?usp=pp_url'
  url += '&entry.865206380=' + mod_id;
  url += '&entry.854684610=' + content;
  url += '&submit=Submit'
  axios.get(url).then(res => {})
  cb(mod_id)
}

// Fetching a json representation of the sheet using axios
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
