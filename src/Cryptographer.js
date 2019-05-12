var CryptoJS = require("crypto-js");


function ReplaceAll(str, search, replacement) {
  var newstr = ''
  str.split(search).map(val => {newstr += val + replacement})
  return newstr.substring(0, newstr.length - replacement.length)
}


function getSpecialsLib(reversed) {
  var lib = {
    '§': '%C2%A7',
    '"': '%22',
    '#': '%23',
    '%': '%25',
    '&': '%26',
    '=': '%3D',
    '`': '%60',
    '^': '%5E',
    '+': '%2B',
    '´': '%C2%B4',
    '¨': '%C2%A8'
  }
  if (reversed) {
    var revlib = {}
    Object.keys(lib).map(key => {
      revlib[lib[key]] = key
    })
    return revlib
  }
  return lib
}


export function Encrypt(data, key) {
  var toEncrypt = {
    test_string: 'there_are_fish_in_the_ocean',
    data: data
  }
  var encrypted = CryptoJS.AES.encrypt(JSON.stringify(toEncrypt), key).toString()
  const specials = getSpecialsLib()
  Object.keys(specials).map(key => {
    encrypted = ReplaceAll(encrypted, key, specials[key])
  })
  return encrypted
}


export function Decrypt(data, key) {
  var msg = data
  const specials = getSpecialsLib(true)
  Object.keys(specials).map(key => {
    msg = ReplaceAll(msg, key, specials[key])
  })
  var decrypted = '';
  try {
    decrypted = CryptoJS.AES.decrypt(msg, key).toString(CryptoJS.enc.Utf8)
  } catch(err) {
    return false
  }
  if (decrypted.includes('there_are_fish_in_the_ocean')) {
    return JSON.parse(decrypted).data
  }
  return false
}