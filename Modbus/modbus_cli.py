import json
import sys
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

def coil_addr(modbus_address):
    """Coil(0xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 1

def discrete_input_addr(modbus_address):
    """Discrete Input(1xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 10001

def input_register_addr(modbus_address):
    """Input Register(3xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 30001

def holding_register_addr(modbus_address):
    """Holding Register(4xxxx) 주소를 내부 인덱스로 변환"""
    return modbus_address - 40001

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
    hr = {}
    ir = {}

    for addr in data:
        if not isinstance(addr, int):
            continue
        reg_type, offset = get_internal_address(addr)
        if reg_type == 'co':
            co[offset] = addr
        elif reg_type == 'di':
            di[offset] = addr
        elif reg_type == 'hr':
            hr[offset] = addr
        elif reg_type == 'ir':
            ir[offset] = addr

    return co, di, hr, ir

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def choose_device(devices):
    print("사용 가능한 장비 목록:")
    for idx, dev_name in enumerate(devices.keys()):
        print(f"{idx + 1}. {dev_name}")
    choice = int(input("접속할 장비 번호 선택: ")) - 1
    dev_name = list(devices.keys())[choice]
    return devices[dev_name]

def modbus_connect(ip, port):
    client = ModbusTcpClient(ip, port=port)
    if not client.connect():
        print("장비에 연결할 수 없습니다.")
        sys.exit(1)

    return client

def load_datamap_json(filename):
    """DataMap JSON 파일을 로드하여 {주소: 값} 형태로 반환"""
    with open(filename, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        return {entry.get("Addr") for entry in raw if entry.get("Addr") is not None}

def user_request_loop(client, device):
    print("\nModbus 기능 코드 목록:")
    print(" 1: Read Coils")
    print(" 2: Read Discrete Inputs")
    print(" 3: Read Holding Registers")
    print(" 4: Read Input Registers")
    print(" 5: Write Single Coil")
    print(" 6: Write Single Holding Register")
    print(" 10: Read All Device Information")
    print(" 0: 장비 선택")

    while True:
        try:
            func_code = int(input("\n기능 코드 입력: "))
        except ValueError:
            print("숫자를 입력하세요.")
            continue

        if func_code == 0:
            break

        try:
            if func_code in {1, 2, 3, 4}:
                address = int(input("주소 입력: "))
                count = int(input("갯수 입력: "))
                if func_code == 1:
                    result = client.read_coils(address=address, count=count, slave=device["unit_id"])
                    print(result.bits[:count])
                elif func_code == 2:
                    result = client.read_discrete_inputs(address=address, count=count, slave=device["unit_id"])
                    print(result.bits[:count])
                elif func_code == 3:
                    result = client.read_holding_registers(address=address, count=count, slave=device["unit_id"])
                    print(result.registers[:count])
                elif func_code == 4:
                    result = client.read_input_registers(address=address, count=count, slave=device["unit_id"])
                    print(result.registers[:count])

                if result.isError():
                    print(f"에러 발생: {result}")

            elif func_code == 5:
                address = int(input("Coil 주소 입력: "))
                value = int(input("값 입력 (0 또는 1): "))
                result = client.write_coil(address, bool(value))
                print("쓰기 결과:", "성공" if not result.isError() else result)

            elif func_code == 6:
                address = int(input("Register 주소 입력: "))
                value = int(input("쓰기 값 입력: "))
                result = client.write_register(address, value)
                print("쓰기 결과:", "성공" if not result.isError() else result)

            elif func_code == 10:
                datamap = load_datamap_json(device["data_map"])
                co, di, hr, ir = create_slave_context(datamap)

                try:
                    for addr in sorted(co):
                        result = client.read_coils(address=addr, count=1, slave=device["unit_id"])
                        print(f"Coil {addr}: {result.bits[0]}")
                    for addr in sorted(di):
                        result = client.read_discrete_inputs(address=addr, count=1, slave=device["unit_id"])
                        print(f"Discrete Input {addr}: {result.bits[0]}")
                    for addr in sorted(hr):
                        result = client.read_holding_registers(address=addr, count=1, slave=device["unit_id"])
                        print(f"Holding Register {addr}: {result.registers[0]}")
                    for addr in sorted(ir):
                        result = client.read_input_registers(address=addr, count=1, slave=device["unit_id"])
                        print(f"Input Register {addr}: {result.registers[0]}")
                except ModbusException as e:
                    print(f"Modbus 오류 발생: {e}")
                    
            else:
                print("잘못된 기능 코드입니다. 다시 입력해주세요.")

        except Exception as e:
            print(f"오류 발생: {e}")

def main():
    file_path = "device_config.json"

    config = load_config(file_path)

    while True:
        device = choose_device(config)
        print(f"\n선택된 장비: {device}")
        client = modbus_connect(device["ip"], device["port"])
        user_request_loop(client, device)
        client.close()
        print("-" * 40)

if __name__ == "__main__":
    main()
