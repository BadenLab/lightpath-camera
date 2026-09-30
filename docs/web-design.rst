======================================
Browser-Server Interaction Patterns
======================================

This document explains how the web browser communicates with the Flask server,
aimed at someone with basic or rusty JavaScript knowledge.

Introduction
============

The camera web interface uses three distinct communication patterns:

1. **MJPEG Streaming** - Continuous video frames over a persistent HTTP connection
2. **Fetch API** - Modern async requests for REST-style operations
3. **Server-Sent Events (SSE)** - Server-initiated push updates

Each pattern solves a different problem, and understanding when to use each
is key to building responsive web applications.


MJPEG Video Streaming
=====================

The simplest pattern requires no JavaScript at all. The video stream is just
an ``<img>`` tag:

.. code-block:: html

    <img id="stream" src="/stream" alt="Camera Stream">

How It Works
------------

When the browser requests ``/stream``, the server responds with a special
MIME type:

.. code-block:: python

    return flask.Response(
        camera.generate_frames(...),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

The key is ``multipart/x-mixed-replace``. This tells the browser:

- Keep the HTTP connection open
- The response contains multiple parts separated by ``--frame``
- Each new part **replaces** the previous one

The server continuously sends JPEG frames::

    --frame
    Content-Type: image/jpeg

    <jpeg bytes>
    --frame
    Content-Type: image/jpeg

    <jpeg bytes>
    ...

The browser handles everything automatically—no JavaScript needed to update
the image. Each frame replaces the previous one, creating the illusion of
video.

Why MJPEG?
----------

- Simple: works in a plain ``<img>`` tag
- Universal: supported by all browsers
- Low latency: frames are pushed immediately
- No JavaScript required

The tradeoff is bandwidth—each frame is a full JPEG image with no inter-frame
compression.


Fetch API for REST Calls
========================

The Fetch API is the modern way to make HTTP requests from JavaScript. All
the button actions (capture, settings) use this pattern.

Async/Await Basics
------------------

JavaScript is single-threaded, so we can't block waiting for network responses.
The ``async/await`` syntax makes asynchronous code look synchronous:

.. code-block:: javascript

    async function refreshSettings() {
        const response = await fetch('/settings');
        const data = await response.json();
        // Use data here...
    }

Key concepts:

- ``async`` marks a function as asynchronous (it returns a Promise)
- ``await`` pauses execution until the Promise resolves
- Code after ``await`` runs when the operation completes
- Errors can be caught with ``try/catch``

Without async/await, you'd write nested callbacks—much harder to read.

GET Requests - Fetching Data
----------------------------

The simplest case retrieves JSON data:

.. code-block:: javascript

    async function refreshSettings() {
        try {
            const response = await fetch('/settings');
            const data = await response.json();
            document.getElementById('exposure').value = data.exposure_time;
            document.getElementById('gain').value = data.analogue_gain;
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }

``fetch('/settings')`` makes an HTTP GET request. The ``response.json()``
method parses the JSON body into a JavaScript object.

POST Requests with Query Parameters
-----------------------------------

To trigger an action (like capturing a photo), use POST:

.. code-block:: javascript

    async function captureRaw() {
        setStatus('Capturing raw...');
        try {
            const response = await fetch('/capture?format=raw', { method: 'POST' });
            if (response.ok) {
                const blob = await response.blob();
                // Handle the binary data...
            } else {
                setStatus('Capture failed: ' + response.statusText);
            }
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }

Key points:

- ``{ method: 'POST' }`` changes from the default GET
- ``?format=raw`` is a query parameter (part of the URL)
- ``response.ok`` is true for status codes 200-299
- ``response.blob()`` reads binary data (for files)

POST Requests with JSON Body
----------------------------

To send structured data to the server:

.. code-block:: javascript

    async function applySettings() {
        const exposure = parseInt(document.getElementById('exposure').value);
        const gain = parseFloat(document.getElementById('gain').value);

        try {
            const response = await fetch('/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exposure_time: exposure, analogue_gain: gain })
            });
            const data = await response.json();
            setStatus(`Applied: exposure=${data.exposure_time}ms`);
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }

Key points:

- ``headers: { 'Content-Type': 'application/json' }`` tells the server we're
  sending JSON
- ``JSON.stringify()`` converts a JavaScript object to a JSON string
- The server receives this in ``flask.request.get_json()``

Query Parameters vs Request Body
--------------------------------

When to use each:

**Query parameters** (``?format=raw``):

- Short, simple values
- Safe to include in URLs (bookmarkable, cacheable)
- Good for filtering or selecting options

**Request body** (JSON):

- Complex or structured data
- Sensitive data (not visible in URL/logs)
- When creating or updating resources


Server-Sent Events (SSE)
========================

SSE solves a different problem: how does the server notify the browser when
something happens?

The Problem
-----------

Consider the homography detection feature. The server processes frames in the
background and finds checkerboard patterns. How do we show results in the UI?

Options:

1. **Polling**: Browser asks "any updates?" every N seconds (wasteful)
2. **WebSockets**: Full bidirectional communication (complex)
3. **SSE**: Server pushes updates when they happen (simple, one-way)

SSE is perfect when you only need server-to-client updates.

EventSource API
---------------

The browser side is simple:

.. code-block:: javascript

    let eventSource = null;

    function startListening() {
        eventSource = new EventSource('/homography/events');

        eventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            updateHomographyDisplay(data);
        };

        eventSource.onerror = () => {
            setStatus('SSE connection error');
        };
    }

    function stopListening() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

Key points:

- ``new EventSource(url)`` opens a persistent connection
- ``onmessage`` fires whenever the server sends data
- ``onerror`` fires on connection problems
- ``close()`` terminates the connection
- The browser auto-reconnects on network errors (built-in!)

Server Side
-----------

The Flask server sends events in a specific format:

.. code-block:: python

    @app.route("/homography/events")
    def homography_events():
        def generate():
            last_timestamp = None
            while True:
                with _homography_lock:
                    result = _homography_result
                if result and result.get("timestamp") != last_timestamp:
                    last_timestamp = result.get("timestamp")
                    yield f"data: {json.dumps(result)}\n\n"
                time.sleep(0.5)

        return flask.Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

Each event is::

    data: {"found": true, "matrix": [...], ...}

    (blank line ends the event)

The ``text/event-stream`` MIME type tells the browser this is an SSE stream.

Why SSE Over WebSockets?
------------------------

- **Simpler**: one-way communication, no handshake protocol
- **Auto-reconnect**: browser handles reconnection automatically
- **HTTP-based**: works through proxies, firewalls, load balancers
- **Sufficient**: when you only need server→client updates

Use WebSockets when you need bidirectional communication (chat apps, games).


Supporting Browser APIs
=======================

The code uses several other JavaScript/browser APIs worth understanding.

DOM Access
----------

``document.getElementById()`` retrieves an HTML element by its ``id`` attribute:

.. code-block:: javascript

    document.getElementById('exposure').value = data.exposure_time;

This finds ``<input id="exposure" ...>`` and sets its value.

CSS Class Manipulation
----------------------

``classList`` manages CSS classes on elements:

.. code-block:: javascript

    panel.classList.add('inactive');     // Add a class
    panel.classList.remove('inactive');  // Remove a class

This is how the UI shows/hides the homography data panel—the ``.inactive``
CSS class sets ``opacity: 0.5``.

Blob Downloads
--------------

To download binary data (images, raw files):

.. code-block:: javascript

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'capture.raw';
    a.click();
    URL.revokeObjectURL(url);

Step by step:

1. ``response.blob()`` reads the response as binary data
2. ``URL.createObjectURL()`` creates a temporary URL pointing to that data
3. Create a hidden ``<a>`` element with the download filename
4. ``a.click()`` triggers the download
5. ``URL.revokeObjectURL()`` frees the memory

Template Literals
-----------------

Backtick strings allow embedded expressions:

.. code-block:: javascript

    setStatus(`Applied: exposure=${data.exposure_time}ms, gain=${data.gain}`);

The ``${...}`` parts are evaluated and inserted into the string. Much cleaner
than string concatenation.


Data Flow Diagrams
==================

Page Load Sequence
------------------

::

    Browser                          Server
       |                               |
       |-------- GET / --------------->|
       |<------- HTML page ------------|
       |                               |
       |-------- GET /stream --------->|
       |<------- MJPEG frames ---------|  (continuous)
       |         (kept open)           |
       |                               |
       |-------- GET /settings ------->|  (from refreshSettings())
       |<------- JSON response --------|
       |                               |
       |-------- GET /config --------->|  (from refreshConfig())
       |<------- JSON response --------|
       |                               |

The page loads, starts the video stream, and fetches initial settings—all
in parallel.

Capture Button Click Flow
-------------------------

::

    User clicks "Capture Raw"
           |
           v
    Browser                          Server
       |                               |
       |--- POST /capture?format=raw ->|
       |                               |  Camera captures...
       |<------ Binary blob -----------|
       |                               |
       v
    Browser creates download
    User receives capture.raw file

Settings Update Flow
--------------------

::

    User changes exposure, clicks "Apply"
           |
           v
    Browser                          Server
       |                               |
       |--- POST /settings ----------->|
       |    Body: {"exposure_time":    |
       |           10000,              |
       |           "analogue_gain":    |
       |           1.5}                |
       |                               |
       |<------ JSON response ---------|
       |        (new settings)         |
       |                               |
       v
    Browser updates status display

SSE Event Flow
--------------

::

    User clicks "Start Detection"
           |
           v
    Browser                          Server
       |                               |
       |--- POST /homography/start --->|
       |<------ {"status":"started"} --|
       |                               |
       |--- GET /homography/events --->|  (SSE connection)
       |                               |
       |        (Server detects checkerboard in background)
       |                               |
       |<------ data: {...} -----------|  (SSE event)
       |                               |
       |        (Browser updates display)
       |                               |
       |<------ data: {...} -----------|  (another event)
       |                               |
       |        (... continues until stopped ...)
       |                               |
    User clicks "Stop Detection"
       |                               |
       |--- POST /homography/stop ---->|
       |<------ {"status":"stopped"} --|
       |                               |
       v
    Browser closes SSE connection
