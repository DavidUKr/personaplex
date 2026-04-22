import { useState, useCallback, useRef } from "react";
import { WSMessage, SocketStatus } from "../../../protocol/types";
import { decodeMessage, encodeMessage } from "../../../protocol/encoder";

export const useSocket = ({
  onMessage,
  uri,
  onDisconnect: onDisconnectProp,
  initialMessages = [],
}: {
  onMessage?: (message: WSMessage) => void;
  uri: string;
  onDisconnect?: () => void;
  initialMessages?: WSMessage[];
}) => {
  const socketRef = useRef<WebSocket | null>(null); // useRef to keep stable socket reference
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("disconnected");

  const sendMessage = useCallback(
    (message: WSMessage) => {
      if (!socketRef.current || socketStatus !== "connected") {
        console.log("socket not connected");
        return;
      }
      socketRef.current.send(encodeMessage(message));
    },
    [socketRef, socketStatus],
  );

  const onConnect = useCallback(() => {
    console.log("connected, now waiting for handshake.");
    setSocketStatus("connecting");
    if (socketRef.current) {
      for (const message of initialMessages) {
        socketRef.current.send(encodeMessage(message));
      }
    }
  }, [initialMessages, setSocketStatus]);

  const onDisconnect = useCallback((event: CloseEvent) => {
    const closedSocket = event.target as WebSocket;
    console.log("disconnected");
    setSocketStatus("disconnected");
    if (onDisconnectProp) {
      onDisconnectProp();
    }
    // ONLY clear socketRef.current if it's the one that closed
    if (socketRef.current === closedSocket) {
      console.log("disconnected (current socket)");
      socketRef.current = null;
      setSocketStatus("disconnected");
      onDisconnectProp?.();
    } else {
      console.log("disconnected (stale socket ignored)");
    }
  }, [onDisconnectProp]);

  const onMessageEvent = useCallback(
    (eventData: MessageEvent) => {
      const dataArray = new Uint8Array(eventData.data);
      const message = decodeMessage(dataArray);
      if (message.type == "handshake") {
        console.log("Handshake received, let's rocknroll.");
        setSocketStatus("connected");
      }
      if (!onMessage) {
        return;
      }
      onMessage(message);
    },
    [onMessage, setSocketStatus],
  );

  const start = useCallback(() => {
    // Close existing socket if any
    if (socketRef.current) {
      console.log("closing existing socket before creating new one");
      socketRef.current.close();
    }
    const ws = new WebSocket(uri);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", onConnect);
    ws.addEventListener("close", onDisconnect);
    ws.addEventListener("message", onMessageEvent);

    socketRef.current = ws;
    console.log("Socket created", ws);
  }, [uri, onMessage, onConnect, onDisconnect, onMessageEvent]);

  const stop = useCallback(() => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      setSocketStatus("disconnected");
      // if (onDisconnectProp) {
      //   onDisconnectProp();
      // }
      // socket?.close();
      // setSocket(null);
  }, []);

  return {
    socketStatus,
    socket: socketRef.current,
    sendMessage,
    start,
    stop,
    setSocketStatus,
  };
};
