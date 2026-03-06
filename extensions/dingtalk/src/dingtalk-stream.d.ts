declare module "dingtalk-stream" {
  export const TOPIC_ROBOT: string;

  export class DWClient {
    constructor(config: {
      clientId?: string;
      clientSecret?: string;
      keepAlive?: boolean;
      debug?: boolean;
    });

    connect(): Promise<void>;
    disconnect?(): Promise<void>;
    registerCallbackListener(
      topic: string,
      callback: (event: {
        headers?: { messageId?: string };
        data?: string;
      }) => Promise<void> | void,
    ): void;
    socketCallBackResponse?(messageId: string, body: unknown): void;
  }
}
