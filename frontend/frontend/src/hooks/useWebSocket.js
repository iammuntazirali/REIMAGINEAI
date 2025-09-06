// import { useEffect, useRef, useState } from 'react';

// export const useWebSocket = url => {
//   const ws = useRef(null);
//   const [messages, setMessages] = useState([]);
//   useEffect(() => {
//     ws.current = new WebSocket(url);
//     ws.current.onmessage = e => setMessages(msgs => [...msgs, JSON.parse(e.data)]);
//     ws.current.onclose = () => setTimeout(() => ws.current = new WebSocket(url), 3000);
//     return () => ws.current.close();
//   }, [url]);
//   return { messages };
// };
import { useEffect, useRef, useState } from 'react';

export const useWebSocket = (url) => {
  const ws = useRef(null);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    ws.current = new WebSocket(url);
    ws.current.onmessage = (event) => setMessages((msgs) => [...msgs, JSON.parse(event.data)]);
    ws.current.onclose = () => setTimeout(() => ws.current = new WebSocket(url), 3000);
    return () => ws.current.close();
  }, [url]);

  return { messages };
};
