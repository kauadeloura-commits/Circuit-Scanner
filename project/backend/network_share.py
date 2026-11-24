"""Choose a LAN address for a phone; never publishes a public tunnel."""
import ipaddress,socket,os
from urllib.parse import urlsplit
from backend.project_export import base_address

def lan_addresses():
    addresses=set()
    try:
        for result in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET):
            ip=ipaddress.ip_address(result[4][0])
            if ip.is_private and not ip.is_loopback and not ip.is_link_local:addresses.add(str(ip))
    except OSError:pass
    return sorted(addresses,key=lambda s:(not s.startswith('192.168.'),not s.startswith('10.'),s))

def share_address(origin):
    address,local=base_address(origin);parts=urlsplit(address)
    choices=lan_addresses();port=parts.port or (443 if parts.scheme=='https' else 80)
    if local:port=int(os.environ.get('CIRCUIT_SCANNER_SHARE_PORT','8012'))
    if local and choices:address=f'http://{choices[0]}:{port}'
    return dict(address=address,alternatives=[f'{parts.scheme}://{ip}:{port}' for ip in choices],
                mobile_ready=not base_address(address)[1],
                message='Use a mesma rede Wi-Fi do computador e mantenha o programa aberto.' if choices or not local else 'Nenhum IP da rede local foi encontrado. Conecte o computador ao Wi-Fi.')
