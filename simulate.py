from shocktube import *
import json
with torch.no_grad():
    left = torch.linspace(-0.5,0,320).cuda()
    dxl = left[1] - left[0]
    right = torch.linspace(0,0.5,40)[1:].cuda()
    dxr = right[1] - right[0]

    left_boundary = torch.linspace(-0.5 - (35 * dxl), -0.5 - dxl, 35).cuda()
    right_boundary = torch.linspace(0.5 + dxr, 0.5 + (35 * dxr), 35).cuda()

    h = 2*(right[1] - right[0])

    left = torch.cat((left_boundary, left), dim=0)
    right = torch.cat((right, right_boundary), dim=0)

    x = torch.cat((left, right), dim=0)

    rho = torch.cat((torch.ones_like(left).cuda(), torch.ones_like(right).cuda()*0.125), dim=0)
    p = torch.cat((torch.ones_like(left).cuda(), torch.ones_like(right).cuda()*0.1), dim=0)
    v = torch.zeros_like(x).cuda()
    gamma = 1.4
    epsilon = 0.5
    eta = 1e-04
    m = 0.0015625000000000

    st = Shocktube(x = x, p = p, rho = rho,
                   v = v, m = m, h = h, gamma = gamma,
                   epsilon = epsilon, eta = eta, kernel = cubic_spline)


    for i in range(2000):
        st.update_euler(dt=1e-04)
        print(i)

    data = dict()
    data['rho'] = st.rho.detach().cpu().to(float).numpy()
    data['p'] = st.p.detach().cpu().to(float).numpy()
    data['v'] = st.v.detach().cpu().to(float).numpy()
    data['x'] = st.x.detach().cpu().to(float).numpy()
    data['e'] = st.e.detach().cpu().to(float).numpy()

    for key in data.keys():
        data[key] = list(data[key])

    with open('shocktube_ce_output.json', 'w') as shock:
        json.dump(data, shock)


    ### Using summation density

    left = torch.linspace(-0.5,0,320).cuda()
    dxl = left[1] - left[0]
    right = torch.linspace(0,0.5,40)[1:].cuda()
    dxr = right[1] - right[0]

    left_boundary = torch.linspace(-0.5 - (35 * dxl), -0.5 - dxl, 35).cuda()
    right_boundary = torch.linspace(0.5 + dxr, 0.5 + (35 * dxr), 35).cuda()

    h = 2*(right[1] - right[0])

    left = torch.cat((left_boundary, left), dim=0)
    right = torch.cat((right, right_boundary), dim=0)

    x = torch.cat((left, right), dim=0)

    rho = torch.cat((torch.ones_like(left).cuda(), torch.ones_like(right).cuda()*0.125), dim=0)
    p = torch.cat((torch.ones_like(left).cuda(), torch.ones_like(right).cuda()*0.1), dim=0)
    v = torch.zeros_like(x).cuda()
    gamma = 1.4
    epsilon = 0.5
    eta = 1e-04
    m = 0.0015625000000000

    st = Shocktube(x = x, p = p, rho = rho, \
                   v = v, m = m, h = h, gamma = gamma, \
                   epsilon = epsilon, eta = eta, kernel = cubic_spline)


    for i in range(2000):
        st.update_euler_SD(dt=1e-04)
        print(i)


    data = dict()
    data['rho'] = st.rho.detach().cpu().to(float).numpy()
    data['p'] = st.p.detach().cpu().to(float).numpy()
    data['v'] = st.v.detach().cpu().to(float).numpy()
    data['x'] = st.x.detach().cpu().to(float).numpy()
    data['e'] = st.e.detach().cpu().to(float).numpy()

    for key in data.keys():
        data[key] = list(data[key])

    with open('shocktube_sd_output.json', 'w') as shock:
        json.dump(data, shock)
