var axios = require('axios');

function rjust(msg, n, rep) {
  msg = '' + msg
  Array.from(Array(Math.max(0, n - msg.length)).keys()).map(() => {
    msg = rep + msg
  })
  return msg
}


function num2bigBase(num, lib) {
  if (!lib) {
    lib = '0123456789'
    lib += 'abcdefghijklmnopqrstuvwxyz'
    lib += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  }
  if (num < lib.length) {
    return lib[num]
  } else {
    return num2bigBase(Math.floor(num / lib.length)) + lib[num % lib.length]
  }
}


function getId(l) {
  l = Math.min(20, Math.max(10, l || 15))
  var lib = '0123456789'
  lib += 'abcdefghijklmnopqrstuvwxyz'
  lib += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  var msg = ''
  msg += Math.round(Math.random() * 100)
  var d = new Date()
  msg += rjust(d.getSeconds(), 2, 0)
  msg += rjust(d.getMinutes(), 2, 0)
  msg += rjust(d.getHours(), 2, 0)
  msg += rjust(d.getDate(), 2, 0)
  msg += rjust(d.getMonth(), 2, 0)
  msg += rjust(parseInt(d.getYear().toString().substring(1)), 2, 0)
  msg = num2bigBase(msg, lib)
  Array.from(Array(l - msg.length).keys()).map(() => {
    msg = lib[Math.round(Math.random() * (lib.length - 1))] + msg
  })
  return msg
}


export function PostToDataBase(data, cb) {
  let model_id = getId(12)
  data.match(/.{1,5000}/g).map((msg, idx) => {
    try {
      var url = 'https://docs.google.com/forms/d/e/1FAIpQLSfA4C6HCBGWLtdfUI4th6VDR7cjtSsj0fp0Tomw96CbgqjO9g/formResponse?usp=pp_url'
      url += '&entry.1116133740=' + model_id;
      url += '&entry.1405886587=' + msg + '_NnUuMmBbEeRr_' + rjust('' + idx, 4, '0');
      url += '&submit=Submit'
      axios.get(url).then(() => {})
    } catch(err) {}
  })
  if (cb) cb(model_id)
}


export function LoadDataBase(cb) {
  axios.get('https://docs.google.com/spreadsheets/d/10LoweeCDvGHQBHb8Plr_1IrLXPy4-N7CFqzcO73MqCs/export?format=csv&id=1qNBuXr5KIHPHqoNBgZEao2F3rAjBtMiQf6fsEDug0mk&gid=0').then(res => {
    var data = {};
    var concat_data = {};
    var raw = res.data.split('\r\n');
    var tmp;
    for (var idx = 1; idx < raw.length; idx++) {
      tmp = raw[idx].split(',')
      if (!(tmp[0] in concat_data)) concat_data[tmp[0]] = {}
      var number = tmp[1].substring(tmp[1].length - 18)
      if (number.includes('NnUuMmBbEeRr')) concat_data[tmp[0]][number] = tmp[1].substring(0, tmp[1].length - 18)
    }
    // Concating data
    Object.keys(concat_data).map(key => {
      data[key] = ''
      Object.keys(concat_data[key]).sort().map(msg => {
        data[key] += concat_data[key][msg]
      })
    })
    if (cb) cb(data)
  })
}