import json
import sys
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

def load_config(file_path="device_config.json"):
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

def user_request_loop(client, unit_id):
    print("\nModbus 기능 코드 목록:")
    print(" 1: Read Coils")
    print(" 2: Read Discrete Inputs")
    print(" 3: Read Holding Registers")
    print(" 4: Read Input Registers")
    print(" 5: Write Single Coil")
    print(" 6: Write Single Holding Register")
    print(" 0: Exit")

    while True:
        try:
            func_code = int(input("\n기능 코드 입력 (0 종료): "))
        except ValueError:
            print("숫자를 입력하세요.")
            continue

        if func_code == 0:
            print("통신 종료")
            break

        try:
            if func_code in {1, 2, 3, 4}:
                address = int(input("주소 입력: "))
                count = int(input("갯수 입력: "))
                if func_code == 1:
                    result = client.read_coils(address=address, count=count, slave=unit_id)
                    print(result.bits[:count])
                elif func_code == 2:
                    result = client.read_discrete_inputs(address=address, count=count, slave=unit_id)
                    print(result.bits[:count])
                elif func_code == 3:
                    result = client.read_holding_registers(address=address, count=count, slave=unit_id)
                    print(result.registers[:count])
                elif func_code == 4:
                    result = client.read_input_registers(address=address, count=count, slave=unit_id)
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

            else:
                print("잘못된 기능 코드입니다. 다시 입력해주세요.")

        except Exception as e:
            print(f"오류 발생: {e}")


    client.close()

def main():
    config = load_config()
    device = choose_device(config)
    print(f"\n선택된 장비: {device}")
    client = modbus_connect(device["ip"], device["port"])
    user_request_loop(client, device["unit_id"])

if __name__ == "__main__":
    main()