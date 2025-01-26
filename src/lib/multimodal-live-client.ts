/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Content, GenerativeContentBlob, Part } from "@google/generative-ai";
import { EventEmitter } from "eventemitter3";
import { difference } from "lodash";
import {
  ClientContentMessage,
  isInterrupted,
  isModelTurn,
  isServerContentMessage,
  isSetupCompleteMessage,
  isToolCallCancellationMessage,
  isToolCallMessage,
  isTurnComplete,
  LiveIncomingMessage,
  ModelTurn,
  RealtimeInputMessage,
  ServerContent,
  SetupMessage,
  StreamingLog,
  ToolCall,
  ToolCallCancellation,
  ToolResponseMessage,
  type LiveConfig,
} from "../multimodal-live-types";
import { blobToJSON, base64ToArrayBuffer } from "./utils";
// Import lamejs as a CommonJS module
const Lame = require('lamejs');

/**
 * the events that this client will emit
 */
interface MultimodalLiveClientEventTypes {
  open: () => void;
  log: (log: StreamingLog) => void;
  close: (event: CloseEvent) => void;
  audio: (data: ArrayBuffer) => void;
  content: (data: ServerContent) => void;
  interrupted: () => void;
  setupcomplete: () => void;
  turncomplete: () => void;
  toolcall: (toolCall: ToolCall) => void;
  toolcallcancellation: (toolcallCancellation: ToolCallCancellation) => void;
  transcription: (text: string) => void;
}

export type MultimodalLiveAPIClientConnection = {
  url?: string;
  apiKey: string;
};

/**
 * A event-emitting class that manages the connection to the websocket and emits
 * events to the rest of the application.
 * If you dont want to use react you can still use this.
 */
export class MultimodalLiveClient extends EventEmitter<MultimodalLiveClientEventTypes> {
  public ws: WebSocket | null = null;
  protected config: LiveConfig | null = null;
  public url: string = "";
  
  // Existing buffer for model audio
  protected audioBuffer: Uint8Array | null = null;

  // NEW buffer for user audio
  protected userAudioBuffer: Uint8Array | null = null;

  public getConfig() {
    return { ...this.config };
  }

  constructor({ url, apiKey }: MultimodalLiveAPIClientConnection) {
    super();
    url =
      url ||
      `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent`;
    url += `?key=${apiKey}`;
    this.url = url;
    this.send = this.send.bind(this);

    // Initialize new user audio buffer
    this.userAudioBuffer = null;
  }

  log(type: string, message: StreamingLog["message"]) {
    const log: StreamingLog = {
      date: new Date(),
      type,
      message,
    };
    this.emit("log", log);
  }

  connect(config: LiveConfig): Promise<boolean> {
    this.config = config;

    const ws = new WebSocket(this.url);

    ws.addEventListener("message", async (evt: MessageEvent) => {
      if (evt.data instanceof Blob) {
        this.receive(evt.data);
      } else {
        console.log("non blob message", evt);
      }
    });
    return new Promise((resolve, reject) => {
      const onError = (ev: Event) => {
        this.disconnect(ws);
        const message = `Could not connect to "${this.url}"`;
        this.log(`server.${ev.type}`, message);
        reject(new Error(message));
      };
      ws.addEventListener("error", onError);
      ws.addEventListener("open", (ev: Event) => {
        if (!this.config) {
          reject("Invalid config sent to `connect(config)`");
          return;
        }
        this.log(`client.${ev.type}`, `connected to socket`);
        this.emit("open");

        this.ws = ws;

        const setupMessage: SetupMessage = {
          setup: this.config,
        };
        this._sendDirect(setupMessage);
        this.log("client.send", "setup");

        ws.removeEventListener("error", onError);
        ws.addEventListener("close", (ev: CloseEvent) => {
          console.log(ev);
          this.disconnect(ws);
          let reason = ev.reason || "";
          if (reason.toLowerCase().includes("error")) {
            const prelude = "ERROR]";
            const preludeIndex = reason.indexOf(prelude);
            if (preludeIndex > 0) {
              reason = reason.slice(
                preludeIndex + prelude.length + 1,
                Infinity,
              );
            }
          }
          this.log(
            `server.${ev.type}`,
            `disconnected ${reason ? `with reason: ${reason}` : ``}`,
          );
          this.emit("close", ev);
        });
        resolve(true);
      });
    });
  }

  disconnect(ws?: WebSocket) {
    // could be that this is an old websocket and theres already a new instance
    // only close it if its still the correct reference
    if ((!ws || this.ws === ws) && this.ws) {
      this.ws.close();
      this.ws = null;
      this.log("client.close", `Disconnected`);
      return true;
    }
    return false;
  }

  protected async receive(blob: Blob) {
    const json = await blobToJSON(blob);
    
    if (isServerContentMessage(json)) {
      const serverContent = json.serverContent;
      this.emit("content", serverContent);

      if (isModelTurn(serverContent)) {
        let textContent = '';

        for (const part of serverContent.modelTurn.parts) {
          // Handle text parts
          if (part.text) {
            textContent += part.text;
          }
          // Handle audio parts
          if (part.inlineData && part.inlineData.mimeType.startsWith("audio/")) {
            const chunkData = base64ToArrayBuffer(part.inlineData.data);
            this.emit("audio", chunkData);
            
            // Accumulate audio data
            if (!this.audioBuffer) {
              this.audioBuffer = new Uint8Array(chunkData);
            } else {
              const newBuffer = new Uint8Array(this.audioBuffer.length + chunkData.byteLength);
              newBuffer.set(this.audioBuffer);
              newBuffer.set(new Uint8Array(chunkData), this.audioBuffer.length);
              this.audioBuffer = newBuffer;
            }
          }
        }

        // If we have text content, log it
        if (textContent) {
          this.log("conversation", `🤖 MODEL: ${textContent}`);
        }
      }

      if (isTurnComplete(serverContent)) {
        this.emit("turncomplete");
        
        // Existing model audio transcription
        if (this.audioBuffer && this.audioBuffer.length > 0) {
          try {
            await this.transcribeAudio(this.audioBuffer, 'model');
          } catch (error) {
            console.error("Failed to transcribe audio:", error);
          } finally {
            this.audioBuffer = null;
          }
        }

        // NEW: transcribe user audio at turn complete
        if (this.userAudioBuffer && this.userAudioBuffer.length > 0) {
          try {
            await this.transcribeAudio(this.userAudioBuffer, 'user');
          } catch (error) {
            console.error("Failed to transcribe user audio:", error);
          } finally {
            this.userAudioBuffer = null;
          }
        }
      }

      if (isInterrupted(serverContent)) {
        this.emit("interrupted");
        this.audioBuffer = null;
      }
    } else if (isToolCallMessage(json)) {
      this.emit("toolcall", json.toolCall);
    } else if (isToolCallCancellationMessage(json)) {
      this.emit("toolcallcancellation", json.toolCallCancellation);
    } else if (isSetupCompleteMessage(json)) {
      this.emit("setupcomplete");
    }
  }

  protected async transcribeAudio(audioData: Uint8Array, source: 'user' | 'model'): Promise<string | null> {
    try {
      if (!audioData || audioData.length === 0) {
        return '<Not recognizable>';
      }

      // Convert PCM to WAV first
      const wavBlob = await this.pcmToWav(audioData);
      
      // Convert WAV to base64
      const base64Audio = await this.blobToBase64(wavBlob);
      
      // Extract API key from the URL
      const apiKey = new URL(this.url).searchParams.get('key');
      if (!apiKey) {
        throw new Error('API key not found');
      }
      
      // Create transcription request using Gemini 1.5 Flash
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: "Generate a transcript of the speech. Please do not include any other text in the response. If you cannot hear the speech, please only say '<Not recognizable>'."
            }, {
              inline_data: {
                mime_type: "audio/wav",
                data: base64Audio
              }
            }]
          }]
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      const transcription = result.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (transcription && transcription !== '<Not recognizable>') {
        const emoji = source === 'user' ? '👤 USER' : '🎙️ TRANSCRIBED';
        this.log("conversation", `${emoji}: ${transcription}`);
      }
      
      return transcription || '<Not recognizable>';

    } catch (error) {
      console.error("Transcription error:", error);
      return '<Not recognizable>';
    }
  }

  protected async pcmToWav(pcmData: Uint8Array): Promise<Blob> {
    // WAV header for 24kHz mono 16-bit audio
    const wavHeader = new ArrayBuffer(44);
    const view = new DataView(wavHeader);
    
    const numChannels = 1;  // Mono
    const sampleRate = 24000;  // 24kHz
    const bitsPerSample = 16;  // 16-bit
    const blockAlign = numChannels * (bitsPerSample / 8);
    const byteRate = sampleRate * blockAlign;
    
    // "RIFF" chunk descriptor
    view.setUint32(0, 0x52494646, false);  // "RIFF" in ASCII
    view.setUint32(4, 36 + pcmData.length, true);  // Total file size
    view.setUint32(8, 0x57415645, false);  // "WAVE" in ASCII

    // "fmt " sub-chunk
    view.setUint32(12, 0x666D7420, false);  // "fmt " in ASCII
    view.setUint32(16, 16, true);  // Length of format data
    view.setUint16(20, 1, true);  // Audio format (1 = PCM)
    view.setUint16(22, numChannels, true);  // Number of channels
    view.setUint32(24, sampleRate, true);  // Sample rate
    view.setUint32(28, byteRate, true);  // Byte rate
    view.setUint16(32, blockAlign, true);  // Block align
    view.setUint16(34, bitsPerSample, true);  // Bits per sample

    // "data" sub-chunk
    view.setUint32(36, 0x64617461, false);  // "data" in ASCII
    view.setUint32(40, pcmData.length, true);  // Data size

    // Create final WAV buffer
    const wavArray = new Uint8Array(wavHeader.byteLength + pcmData.length);
    wavArray.set(new Uint8Array(wavHeader));
    wavArray.set(pcmData, wavHeader.byteLength);

    return new Blob([wavArray], { type: 'audio/wav' });
  }

  protected async blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (typeof reader.result === 'string') {
          // Remove the "data:audio/wav;base64," prefix
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } else {
          reject(new Error('Failed to convert blob to base64'));
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * send realtimeInput, this is base64 chunks of "audio/pcm" and/or "image/jpg"
   */
  sendRealtimeInput(chunks: GenerativeContentBlob[]) {
    for (const ch of chunks) {
      if (ch.mimeType.includes("audio")) {
        // Convert base64 to raw PCM
        const rawData: string = atob(ch.data);
        const rawBuffer: Uint8Array = new Uint8Array(rawData.length);
        for (let i = 0; i < rawData.length; i++) {
          rawBuffer[i] = rawData.charCodeAt(i);
        }

        // Accumulate user audio
        if (!this.userAudioBuffer) {
          this.userAudioBuffer = rawBuffer;
        } else {
          const newBuffer: Uint8Array = new Uint8Array(
            this.userAudioBuffer.length + rawBuffer.length
          );
          newBuffer.set(this.userAudioBuffer);
          newBuffer.set(rawBuffer, this.userAudioBuffer.length);
          this.userAudioBuffer = newBuffer;
        }
      }
    }

    // Send the realtime input message to the server
    const data: RealtimeInputMessage = {
      realtimeInput: {
        mediaChunks: chunks,
      },
    };
    this._sendDirect(data);
  }

  /**
   *  send a response to a function call and provide the id of the functions you are responding to
   */
  sendToolResponse(toolResponse: ToolResponseMessage["toolResponse"]) {
    const message: ToolResponseMessage = {
      toolResponse,
    };

    this._sendDirect(message);
    this.log(`client.toolResponse`, message);
  }

  /**
   * send normal content parts such as { text }
   */
  send(parts: Part | Part[], turnComplete: boolean = true) {
    parts = Array.isArray(parts) ? parts : [parts];
    const content: Content = {
      role: "user",
      parts,
    };

    // Log user's text input if present
    for (const part of parts) {
      if ('text' in part && part.text) {
        this.log("user.text", `👤 USER SAYS: ${part.text}`);
      }
    }

    const clientContentRequest: ClientContentMessage = {
      clientContent: {
        turns: [content],
        turnComplete,
      },
    };

    this._sendDirect(clientContentRequest);
  }

  /**
   *  used internally to send all messages
   *  don't use directly unless trying to send an unsupported message type
   */
  _sendDirect(request: object) {
    if (!this.ws) {
      throw new Error("WebSocket is not connected");
    }
    const str = JSON.stringify(request);
    this.ws.send(str);
  }
}

