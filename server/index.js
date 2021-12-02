const http = require('http')
const path = require('path')
const express = require('express')
const socketIo = require('socket.io')
const needle = require('needle')
const config = require('dotenv').config()
const TOKEN = process.env.TWITTER_BEARER_TOKEN
const PORT = process.env.PORT || 3000

const app = express()

const server = http.createServer(app)
const io = socketIo(server)

app.get('/', (req, res) => {
  res.sendFile(path.resolve(__dirname, '../', 'client', 'index.html'))
})

const rulesURL = 'https://api.twitter.com/2/tweets/sample/stream'
const streamURL =
  'https://api.twitter.com/2/tweets/sample/stream?tweet.fields=public_metrics&expansions=author_id'

function streamTweets(socket) {
  const stream = needle.get(streamURL, {
    headers: {
      Authorization: `Bearer ${TOKEN}`,
    },
  })

  stream.on('data', (data) => {
    try {
      const json = JSON.parse(data)
      console.log(json)
      socket.emit('tweet', json)
    } catch (error) {}
  })

  return stream
}

io.on('connection', async () => {
  console.log('Client connected...')

  

  const filteredStream = streamTweets(io)

  let timeout = 0
  filteredStream.on('timeout', () => {
    // Reconnect on error
    console.warn('A connection error occurred. Reconnecting…')
    setTimeout(() => {
      timeout++
      streamTweets(io)
    }, 2 ** timeout)
    streamTweets(io)
  })
})

server.listen(PORT, () => console.log(`Listening on port ${PORT}`))