import json
import signal
import sys
import time
import os
from threading import Thread
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext
from pymodbus.datastore.store import ModbusSequentialDataBlock

# Modbus 주소를 내부 인덱스로 변환하는 함수들

def coil_addr(modbus_address):
    """Coil(0xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address

def discrete_input_addr(modbus_address):
    """Discrete Input(1xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 10000

def input_register_addr(modbus_address):
    """Input Register(3xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 30000

def holding_register_addr(modbus_address):
    """Holding Register(4xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 40000

def get_internal_address(addr):
    """입력된 Modbus 주소에 따라 레지스터 타입과 내부 인덱스를 반환"""
    if 1 <= addr <= 9999:
        return 'co', coil_addr(addr)
    elif 10001 <= addr <= 19999:
        return 'di', discrete_input_addr(addr)
    elif 30001 <= addr <= 39999:
        return 'ir', input_register_addr(addr)
    elif 40001 <= addr <= 49999:
        return 'hr', holding_register_addr(addr)
    else:
        raise ValueError(f"지원하지 않는 Modbus 주소 범위: {addr}")

def create_slave_context(data):
    """레지스터 타입별로 데이터를 나누고, Modbus 서버 컨텍스트를 생성"""
    co = {}
    di = {}
    ir = {}
    hr = {}

    for addr, value in data.items():
        if not isinstance(addr, int):
            continue
        reg_type, offset = get_internal_address(addr)
        if reg_type == 'co':
            co[offset] = value
        elif reg_type == 'di':
            di[offset] = value
        elif reg_type == 'ir':
            ir[offset] = value
        elif reg_type == 'hr':
            hr[offset] = value

    def create_block(data_map):
        """데이터 맵에서 SequentialDataBlock 생성"""
        return ModbusSequentialDataBlock(0, [data_map.get(i, 0) for i in range(max(data_map.keys()) + 1 if data_map else 1)])

    store = ModbusSlaveContext(
        co=create_block(co),
        di=create_block(di),
        ir=create_block(ir),
        hr=create_block(hr)
    )
    return ModbusServerContext(slaves=store, single=True)

def start_modbus_device(context, ip, port):
    """Modbus TCP 서버 실행"""
    StartTcpServer(context, address=(ip, port))

def signal_handler(sig, frame):
    """Ctrl+C 시 서버 종료 처리"""
    print("\n🛑 서버 종료 중...")
    sys.exit(0)

def load_device_json(filename):
    """장비 구성 파일(JSON)을 불러와 딕셔너리로 반환"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_datamap_json(filename):
    """DataMap JSON 파일을 로드하여 {주소: 값} 형태로 반환"""
    with open(filename, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        return {entry.get("Addr"): entry.get("Value", 0) for entry in raw if entry.get("Addr") is not None}

def main(devices):
    """각 장비에 대해 Modbus 서버 스레드를 실행하고 대기"""
    signal.signal(signal.SIGINT, signal_handler)

    for device, config in devices.items():
        datamap = load_datamap_json(config["data_map"])
        context = create_slave_context(datamap)
        thread = Thread(target=start_modbus_device, args=(context, config["ip"], config["port"]), daemon=True)
        thread.start()

        print(f"🔌 {device} 장비 서버가 실행 중입니다. [IP: {config['ip']}, PORT: {config['port']}]")

    print("\n연결 종료 시 Ctrl+C를 눌러주세요.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    device_file = "device_config.json"
    devices = load_device_json(device_file)
    main(devices)