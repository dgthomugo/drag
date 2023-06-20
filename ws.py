import json
from websocket_server import WebsocketServer


class Websocket:
    def __init__(self):
        self.ball_position = False
        self.is_runing = True
        
    # Called for every client connecting (after handshake)
    def new_client(self, client, server):
        print("New client connected and was given id %d" % client['id'])
    
    # Called for every client disconnecting
    def client_left(self, client, server):
        print("Client(%d) disconnected" % client['id'])
        
    def send_message(self, message):
        msg = json.dumps(message)
        print(msg)
        self.server.send_message_to_all(msg)

    def newServer(self, host, port):
        # 创建Websocket Server
        self.server = WebsocketServer(host=host, port=port)
        # 有设备连接上了
        self.server.set_fn_new_client(self.new_client)
        # 断开连接
        self.server.set_fn_client_left(self.client_left)
        # 接收到信息
        # server.set_fn_message_received(message_received)
        # 开始监听
        self.server.run_forever()
            
    def stop(self):
        self.is_runing = False
        self.server.shutdown()

    