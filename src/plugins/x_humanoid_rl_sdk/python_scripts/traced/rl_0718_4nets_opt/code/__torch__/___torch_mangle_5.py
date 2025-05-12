class actor_twoexperts(Module):
  __parameters__ = []
  __buffers__ = []
  def forward(self: __torch__.___torch_mangle_5.actor_twoexperts,
    obs: Tensor,
    gait_a: float) -> Tensor:
    obs_clipped = torch.clamp(obs, -100., 100.)
    if torch.eq(gait_a, 0.):
      _0 = torch.matmul(obs_clipped, CONSTANTS.c0)
      input = torch.elu(torch.add(_0, CONSTANTS.c1), 1.)
      _1 = torch.add(torch.matmul(input, CONSTANTS.c2), CONSTANTS.c3)
      input0 = torch.elu(_1, 1.)
      _2 = torch.matmul(input0, CONSTANTS.c4)
      input1 = torch.elu(torch.add(_2, CONSTANTS.c5), 1.)
      _3 = torch.matmul(input1, CONSTANTS.c6)
      actions = torch.add(_3, CONSTANTS.c7)
    else:
      if torch.eq(gait_a, 1.):
        _4 = torch.matmul(obs_clipped, CONSTANTS.c8)
        input2 = torch.elu(torch.add(_4, CONSTANTS.c9), 1.)
        _5 = torch.matmul(input2, CONSTANTS.c10)
        input3 = torch.elu(torch.add(_5, CONSTANTS.c11), 1.)
        _6 = torch.matmul(input3, CONSTANTS.c12)
        input4 = torch.elu(torch.add(_6, CONSTANTS.c13), 1.)
        _7 = torch.matmul(input4, CONSTANTS.c14)
        actions0 = torch.add(_7, CONSTANTS.c15)
      else:
        if torch.eq(gait_a, 2.):
          _8 = torch.matmul(obs_clipped, CONSTANTS.c16)
          input5 = torch.elu(torch.add(_8, CONSTANTS.c17), 1.)
          _9 = torch.matmul(input5, CONSTANTS.c18)
          input6 = torch.elu(torch.add(_9, CONSTANTS.c19), 1.)
          _10 = torch.matmul(input6, CONSTANTS.c20)
          input7 = torch.elu(torch.add(_10, CONSTANTS.c21), 1.)
          _11 = torch.matmul(input7, CONSTANTS.c22)
          actions1 = torch.add(_11, CONSTANTS.c23)
        else:
          if torch.eq(gait_a, 3.):
            _12 = torch.matmul(obs_clipped, CONSTANTS.c24)
            _13 = torch.add(_12, CONSTANTS.c25)
            input8 = torch.elu(_13, 1.)
            _14 = torch.matmul(input8, CONSTANTS.c26)
            _15 = torch.add(_14, CONSTANTS.c27)
            input9 = torch.elu(_15, 1.)
            _16 = torch.matmul(input9, CONSTANTS.c28)
            _17 = torch.add(_16, CONSTANTS.c29)
            input10 = torch.elu(_17, 1.)
            _18 = torch.matmul(input10, CONSTANTS.c30)
            _19 = torch.add(_18, CONSTANTS.c31)
            actions2 = _19
          else:
            actions2 = CONSTANTS.c32
          actions1 = actions2
        actions0 = actions1
      actions = actions0
    return actions
