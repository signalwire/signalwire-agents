"""Protocol constants for the SignalWire RELAY calling API."""

# Protocol version
PROTOCOL_VERSION = {"major": 2, "minor": 0, "revision": 0}
AGENT_STRING = "signalwire-agents-python/1.0"

# JSON-RPC methods
METHOD_SIGNALWIRE_CONNECT = "signalwire.connect"
METHOD_SIGNALWIRE_EVENT = "signalwire.event"
METHOD_SIGNALWIRE_PING = "signalwire.ping"
METHOD_SIGNALWIRE_DISCONNECT = "signalwire.disconnect"
METHOD_SIGNALWIRE_RECEIVE = "signalwire.receive"
METHOD_SIGNALWIRE_UNRECEIVE = "signalwire.unreceive"

# Authorization state event
EVENT_AUTHORIZATION_STATE = "signalwire.authorization.state"

# Call states
CALL_STATE_CREATED = "created"
CALL_STATE_RINGING = "ringing"
CALL_STATE_ANSWERED = "answered"
CALL_STATE_ENDING = "ending"
CALL_STATE_ENDED = "ended"

CALL_STATES = (
    CALL_STATE_CREATED,
    CALL_STATE_RINGING,
    CALL_STATE_ANSWERED,
    CALL_STATE_ENDING,
    CALL_STATE_ENDED,
)

# End reasons
END_REASON_HANGUP = "hangup"
END_REASON_CANCEL = "cancel"
END_REASON_BUSY = "busy"
END_REASON_NO_ANSWER = "noAnswer"
END_REASON_DECLINE = "decline"
END_REASON_ERROR = "error"
END_REASON_ABANDONED = "abandoned"
END_REASON_MAX_DURATION = "max_duration"
END_REASON_NOT_FOUND = "not_found"

# Connect states
CONNECT_STATE_CONNECTING = "connecting"
CONNECT_STATE_CONNECTED = "connected"
CONNECT_STATE_DISCONNECTED = "disconnected"
CONNECT_STATE_FAILED = "failed"

# Event types
EVENT_CALL_STATE = "calling.call.state"
EVENT_CALL_RECEIVE = "calling.call.receive"
EVENT_CALL_CONNECT = "calling.call.connect"
EVENT_CALL_PLAY = "calling.call.play"
EVENT_CALL_COLLECT = "calling.call.collect"
EVENT_CALL_RECORD = "calling.call.record"
EVENT_CALL_DETECT = "calling.call.detect"
EVENT_CALL_FAX = "calling.call.fax"
EVENT_CALL_TAP = "calling.call.tap"
EVENT_CALL_SEND_DIGITS = "calling.call.send_digits"
EVENT_CALL_DIAL = "calling.call.dial"
EVENT_CALL_REFER = "calling.call.refer"
EVENT_CALL_DENOISE = "calling.call.denoise"
EVENT_CALL_PAY = "calling.call.pay"
EVENT_CALL_QUEUE = "calling.call.queue"
EVENT_CALL_STREAM = "calling.call.stream"
EVENT_CALL_ECHO = "calling.call.echo"
EVENT_CALL_TRANSCRIBE = "calling.call.transcribe"
EVENT_CONFERENCE = "calling.conference"
EVENT_CALLING_ERROR = "calling.error"

# Messaging event types
EVENT_MESSAGING_RECEIVE = "messaging.receive"
EVENT_MESSAGING_STATE = "messaging.state"

# Message states
MESSAGE_STATE_QUEUED = "queued"
MESSAGE_STATE_INITIATED = "initiated"
MESSAGE_STATE_SENT = "sent"
MESSAGE_STATE_DELIVERED = "delivered"
MESSAGE_STATE_UNDELIVERED = "undelivered"
MESSAGE_STATE_FAILED = "failed"
MESSAGE_STATE_RECEIVED = "received"

MESSAGE_TERMINAL_STATES = (
    MESSAGE_STATE_DELIVERED,
    MESSAGE_STATE_UNDELIVERED,
    MESSAGE_STATE_FAILED,
)

# Play states
PLAY_STATE_PLAYING = "playing"
PLAY_STATE_PAUSED = "paused"
PLAY_STATE_FINISHED = "finished"
PLAY_STATE_ERROR = "error"

# Record states
RECORD_STATE_RECORDING = "recording"
RECORD_STATE_PAUSED = "paused"
RECORD_STATE_FINISHED = "finished"
RECORD_STATE_NO_INPUT = "no_input"

# Detect types
DETECT_TYPE_MACHINE = "machine"
DETECT_TYPE_FAX = "fax"
DETECT_TYPE_DIGIT = "digit"

# Join room states
ROOM_STATE_JOINING = "joining"
ROOM_STATE_JOIN = "join"
ROOM_STATE_LEAVING = "leaving"
ROOM_STATE_LEAVE = "leave"

# Reconnect settings
RECONNECT_MIN_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
RECONNECT_BACKOFF_FACTOR = 2.0

# Default host
DEFAULT_RELAY_HOST = "relay.signalwire.com"
