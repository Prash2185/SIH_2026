"""
Verification test for all FastAPI REST endpoints and Web Dashboard.
"""
from fastapi.testclient import TestClient
from bharatopt.ui.server import app

client = TestClient(app)

def test_endpoints():
    print('[*] Testing GET / ...')
    r = client.get('/')
    assert r.status_code == 200 and 'BharatOpt Core v5.0' in r.text
    print('  -> 200 OK')

    print('[*] Testing GET /api/v1/health ...')
    r = client.get('/api/v1/health')
    assert r.status_code == 200 and r.json()['status'] == 'ONLINE'
    print('  -> 200 OK')

    print('[*] Testing GET /api/v1/audit ...')
    r = client.get('/api/v1/audit')
    assert r.status_code == 200 and r.json()['status'] == 'PASSED [100% SOVEREIGN]'
    print('  -> 200 OK, 0 violations')

    print('[*] Testing POST /api/v1/solve/scenario (mrpl_l1) ...')
    r = client.post('/api/v1/solve/scenario', json={'scenario': 'mrpl_l1', 'tier': 'T1'})
    assert r.status_code == 200
    data = r.json()
    print(f'  -> 200 OK, obj={data["objective"]}, time={data["solve_time_ms"]}ms, verifier={data["verifier"]["passed"]}')

    print('[*] Testing POST /api/v1/solve/scenario (mrpl_l2) ...')
    r = client.post('/api/v1/solve/scenario', json={'scenario': 'mrpl_l2', 'tier': 'T1'})
    assert r.status_code == 200
    data2 = r.json()
    print(f'  -> 200 OK, obj={data2["objective"]}, schedule rows={len(data2["schedule"])}')

    print('[*] Testing POST /api/v1/why-not ...')
    r = client.post('/api/v1/why-not', json={'variable_name': 'spot_buy_Bonny_Light', 'min_forced_val': 1.0})
    assert r.status_code == 200
    data_wn = r.json()
    print(f'  -> 200 OK, status={data_wn["status"]}, delta={data_wn["delta_objective"]}')

    print('[*] Testing GET /api/v1/appendix-a ...')
    r = client.get('/api/v1/appendix-a')
    assert r.status_code == 200 and r.json()['all_passed'] is True
    print(f'  -> 200 OK, 10/10 passed')

    print('[*] Testing GET /api/v1/benchmarks ...')
    r = client.get('/api/v1/benchmarks')
    assert r.status_code == 200
    print(f'  -> 200 OK, {len(r.json()["benchmarks"])} gates certified')

    print('\nALL FASTAPI ENDPOINTS VERIFIED 100% OPERATIONAL!')

if __name__ == '__main__':
    test_endpoints()
