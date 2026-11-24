"""Start once; subsequent launches open the running Circuit Scanner."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
from urllib.request import urlopen

LAUNCHER_VERSION='5.44.2'

class InstanceLock:
    """An OS lock is released even if the process is interrupted or crashes."""
    def __init__(self,path):self.path=Path(path);self.file=None
    def acquire(self):
        self.file=self.path.open('a+b')
        if self.path.stat().st_size==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();self.file=None;return False
        return True
    def close(self):
        if self.file is not None:self.file.close();self.file=None

def scanner_running(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/system/version',timeout=2) as response:
            data=json.loads(response.read(32768))
        # Recognize v5.44.1 without stopping the user's current analysis.
        return isinstance(data,dict) and (data.get('application')=='circuit-scanner' or
            isinstance(data.get('build'),str) and data.get('local_falstad_api') is True
            and data.get('spatial_connected_schematic') is True)
    except (OSError,ValueError):return False

def reserve_socket(host,port):
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        if os.name=='nt':sock.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        else:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind((host,port));sock.listen(128);sock.setblocking(False)
        return sock
    except BaseException:
        sock.close();raise

def open_running(url,*,browser=True,starting=False):
    print('O Circuit Scanner já está iniciando ou aberto.' if starting else 'O Circuit Scanner já está aberto.')
    print(f'Abra no computador: {url}')
    if browser:
        import webbrowser
        try:webbrowser.open(url)
        except OSError:pass
    return 0

def serve(editor_socket,shared_socket,port,mobile):
    # Heavy imports follow the lock and both reservations. A double-click
    # cannot load two copies of the models or start half of the application.
    import uvicorn
    from backend.main import app
    from backend.mobile_share import mobile_app
    shared=uvicorn.Server(uvicorn.Config(mobile_app(app),host='0.0.0.0',port=mobile,log_level='warning'))
    worker=threading.Thread(target=shared.run,kwargs={'sockets':[shared_socket]},daemon=True)
    local=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port))
    worker.start()
    print(f'Abra no computador: http://127.0.0.1:{port}/workflow')
    print(f'O QR Code usa a porta {mobile}, disponível para consulta na mesma rede Wi-Fi.')
    try:local.run(sockets=[editor_socket])
    finally:shared.should_exit=True;worker.join(timeout=3)

def main(argv=None):
    parser=argparse.ArgumentParser(description='Abrir o Circuit Scanner')
    parser.add_argument('--no-browser',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:
        port=int(os.environ.get('CIRCUIT_SCANNER_PORT','8010'))
        mobile=int(os.environ.get('CIRCUIT_SCANNER_SHARE_PORT','8012'))
        if not(1<=port<=65535 and 1<=mobile<=65535) or port==mobile:raise ValueError
    except ValueError:
        print('As portas do programa e do compartilhamento devem ser números diferentes entre 1 e 65535.');return 1
    url=f'http://127.0.0.1:{port}/workflow'
    if scanner_running(port):return open_running(url,browser=not args.no_browser)
    guard=InstanceLock(Path(__file__).resolve().parent/f'.scanner_{port}.lock')
    sockets=[]
    try:
        if not guard.acquire():return open_running(url,browser=not args.no_browser,starting=True)
        for host,number in (('127.0.0.1',port),('0.0.0.0',mobile)):
            try:sockets.append(reserve_socket(host,number))
            except OSError:
                if number==port and scanner_running(port):return open_running(url,browser=not args.no_browser)
                label='do programa' if number==port else 'do compartilhamento no celular'
                print(f'A porta {number} {label} está ocupada por outro programa. Feche esse programa ou configure uma porta livre.');return 1
        os.environ['CIRCUIT_SCANNER_SHARE_PORT']=str(mobile)
        serve(sockets[0],sockets[1],port,mobile)
        return 0
    except KeyboardInterrupt:return 0
    except OSError as error:
        print(f'Não foi possível iniciar o Circuit Scanner: {error}');return 1
    finally:
        for sock in sockets:sock.close()
        guard.close()

if __name__=='__main__':sys.exit(main())
