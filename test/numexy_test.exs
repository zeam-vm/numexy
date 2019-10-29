defmodule NumexyTest do
  alias Numexy.Array
  use ExUnit.Case
  doctest Numexy

  test "vector struct" do
    v = Numexy.new([1, 2, 3])
    assert v.array == [1, 2, 3]
    assert v.shape == [3]
  end

  test "matrix struct" do
    m = Numexy.new([[1, 2, 3], [1, 2, 3]])
    assert m.array == [[1, 2, 3], [1, 2, 3]]
    assert m.shape == [2, 3]
  end

  test "add vector and scalar" do
    x = Numexy.new([1, 2, 3])
    y = 4
    v = Numexy.add(x, y)
    assert v.array == [5, 6, 7]
    assert v.shape == [3]
    v = Numexy.add(y, x)
    assert v.array == [5, 6, 7]
    assert v.shape == [3]
  end

  test "add vector and vector" do
    x = Numexy.new([1, 2, 3])
    y = Numexy.new([4, 5, 2])
    v = Numexy.add(x, y)
    assert v.array == [5, 7, 5]
    assert v.shape == [3]
  end

  test "add matrix and scalar" do
    x = Numexy.new([[1, 2, 3], [4, 5, 6]])
    y = 4
    m = Numexy.add(x, y)
    assert m.array == [[5, 6, 7], [8, 9, 10]]
    assert m.shape == [2, 3]
    x = 4
    y = Numexy.new([[1, 2, 3], [4, 5, 6]])
    m = Numexy.add(x, y)
    assert m.array == [[5, 6, 7], [8, 9, 10]]
    assert m.shape == [2, 3]
  end

  test "add matrix and matrix" do
    x = Numexy.new([[1, 2, 3], [4, 5, 6]])
    y = Numexy.new([[4, 5, 2], [1, 7, 3]])
    m = Numexy.add(x, y)
    assert m.array == [[5, 7, 5], [5, 12, 9]]
    assert m.shape == [2, 3]
  end

  test "sub vector and scalar" do
    x = Numexy.new([1, 2, 3])
    y = 4
    v = Numexy.sub(x, y)
    assert v.array == [-3, -2, -1]
    assert v.shape == [3]
    v = Numexy.sub(y, x)
    assert v.array == [3, 2, 1]
    assert v.shape == [3]
  end

  test "sub vector and vector" do
    x = Numexy.new([1, 2, 3])
    y = Numexy.new([4, 5, 2])
    v = Numexy.sub(x, y)
    assert v.array == [-3, -3, 1]
    assert v.shape == [3]
  end

  test "sub matrix and scalar" do
    x = Numexy.new([[1, 2, 3], [4, 5, 6]])
    y = 4
    m = Numexy.sub(x, y)
    assert m.array == [[-3, -2, -1], [0, 1, 2]]
    assert m.shape == [2, 3]
    x = 4
    y = Numexy.new([[1, 2, 3], [4, 5, 6]])
    m = Numexy.sub(x, y)
    assert m.array == [[3, 2, 1], [0, -1, -2]]
    assert m.shape == [2, 3]
  end

  test "sub matrix and matrix" do
    x = Numexy.new([[1, 2, 3], [4, 5, 6]])
    y = Numexy.new([[4, 5, 2], [1, 7, 3]])
    m = Numexy.sub(x, y)
    assert m.array == [[-3, -3, 1], [3, -2, 3]]
    assert m.shape == [2, 3]
  end

  test "multiplication vector and scalar" do
    x = Numexy.new([1, 2, 3])
    y = 3
    v = Numexy.mul(x, y)
    assert v.array == [3, 6, 9]
    assert v.shape == [3]
  end

  test "multiplication vector and vector" do
    x = Numexy.new([1, 2, 3])
    y = Numexy.new([4, 5, 2])
    v = Numexy.mul(x, y)
    assert v.array == [4, 10, 6]
    assert v.shape == [3]
  end

  test "multiplication matrix and scalar" do
    x = Numexy.new([[1, 2, 3], [4, 5, 6]])
    y = 4
    m = Numexy.mul(x, y)
    assert m.array == [[4, 8, 12], [16, 20, 24]]
    assert m.shape == [2, 3]
    x = 4
    y = Numexy.new([[1, 2, 3], [4, 5, 6]])
    m = Numexy.mul(x, y)
    assert m.array == [[4, 8, 12], [16, 20, 24]]
    assert m.shape == [2, 3]
  end

  test "multiplication matrix and matrix" do
    x = Numexy.new([[1, 2, 3], [4, 5, 6]])
    y = Numexy.new([[4, 5, 2], [1, 7, 3]])
    m = Numexy.mul(x, y)
    assert m.array == [[4, 10, 6], [4, 35, 18]]
    assert m.shape == [2, 3]
  end

  test "div vector and scalar" do
    x = Numexy.new([9, 6, 3])
    y = 3
    v = Numexy.div(x, y)
    assert v.array == [3.0, 2.0, 1.0]
    assert v.shape == [3]
  end

  test "div vector and vector" do
    x = Numexy.new([8, 10, 4])
    y = Numexy.new([4, 5, 2])
    v = Numexy.div(x, y)
    assert v.array == [2.0, 2.0, 2.0]
    assert v.shape == [3]
  end

  test "div matrix and scalar" do
    x = Numexy.new([[8, 4, 4], [4, 12, 8]])
    y = 4
    m = Numexy.div(x, y)
    assert m.array == [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0]]
    assert m.shape == [2, 3]
    x = 4
    y = Numexy.new([[8, 4, 4], [4, 12, 8]])
    m = Numexy.div(x, y)
    assert m.array == [[0.5, 1.0, 1.0], [1.0, 1.0/3.0, 0.5]]
    assert m.shape == [2, 3]
  end

  test "div matrix and matrix" do
    x = Numexy.new([[8, 10, 6], [4, 7, 6]])
    y = Numexy.new([[4, 5, 2], [1, 7, 3]])
    m = Numexy.div(x, y)
    assert m.array == [[2.0, 2.0, 3.0], [4.0, 1.0, 2.0]]
    assert m.shape == [2, 3]
  end

  test "inner product (vector and vector)" do
    x = Numexy.new([1, 2, 3])
    y = Numexy.new([1, 2, 3])
    assert Numexy.dot(x, y) == 14
  end

  test "inner product (matrix and vector)" do
    x = Numexy.new([[1, 6, 4], [2, 9, 5]])
    y = Numexy.new([1, 2, 3])
    m = Numexy.dot(x, y)
    assert m.array == [25, 35]
    assert m.shape == [2]
  end

  test "inner product (matrix and matrix)" do
    x = Numexy.new([[1, 6, 4], [2, 9, 5]])
    y = Numexy.new([[4, 3], [7, 5], [2, 7]])
    m = Numexy.dot(x, y)
    assert m.array == [[54, 61], [81, 86]]
    assert m.shape == [2, 2]
  end

  test "transpose matrix." do
    x = Numexy.new([[4, 3], [7, 5], [2, 7]])
    m = Numexy.transpose(x)
    assert m.array == [[4, 7, 2], [3, 5, 7]]
    assert m.shape == [2, 3]
  end

  test "matrix ones." do
    m = Numexy.ones([2, 3])
    assert m.array == [[1, 1, 1], [1, 1, 1]]
    assert m.shape == [2, 3]
  end

  test "vector ones." do
    m = Numexy.ones([3])
    assert m.array == [1, 1, 1]
    assert m.shape == [3]
  end

  test "matrix zeros." do
    m = Numexy.zeros([2, 3])
    assert m.array == [[0, 0, 0], [0, 0, 0]]
    assert m.shape == [2, 3]
  end

  test "vector zeros." do
    m = Numexy.zeros([3])
    assert m.array == [0, 0, 0]
    assert m.shape == [3]
  end

  test "vector sum." do
    v = Numexy.new([2, 9, 5])
    assert 16 == Numexy.sum(v)
  end

  test "matrix sum." do
    m = Numexy.new([[1, 2, 3], [4, 5, 6]])
    assert 21 == Numexy.sum(m)
  end

  test "vector avarage." do
    v = Numexy.new([2, 9, 5])
    assert 5.333333333333333 == Numexy.avg(v)
  end

  test "matrix avarage." do
    m = Numexy.new([[1, 2, 3], [4, 5, 6]])
    assert 3.5 == Numexy.avg(m)
  end

  test "vector get value" do
    v = Numexy.new([2, 9, 5])
    assert 9 == Numexy.get(v, [2])
  end

  test "test no arguments argmax." do
    m = Numexy.new([[1, 2, 9], [4, 5, 6]])
    assert 2 == Numexy.argmax(m)
  end

  test "test :row argument argmax." do
    m = Numexy.new([[1, 2, 9], [4, 6, 3]])
    assert [2, 1] == Numexy.argmax(m, :row)
  end

  test "test :col argument argmax." do
    m = Numexy.new([[1, 2, 9], [4, 6, 3]])
    assert [1, 1, 0] == Numexy.argmax(m, :col)
  end

  test "test step function" do
    v = Numexy.new([-2, 9, 5]) |> Numexy.step_function()
    assert v.array == [0, 1, 1]
  end

  test "test sigmoid function" do
    v = Numexy.new([-2, 9, 5]) |> Numexy.sigmoid()
    assert v.array == [0.11920292202211755, 0.9998766054240137, 0.9933071490757153]
  end

  test "test relu function" do
    v = Numexy.new([-2, 9, 5]) |> Numexy.relu()
    assert v.array == [0, 9, 5]
  end

  test "test softmax function" do
    v = Numexy.new([-2, 9, 5]) |> Numexy.softmax()
    assert v.array == [1.6401031494862326e-5, 0.9819976839988096, 0.017985914969695496]
  end

  test "test reshape" do
    m = Numexy.reshape([1, 2, 3, 4, 5, 6], 3)
    assert m.array == [[1, 2, 3], [4, 5, 6]]
    assert m.shape == [2, 3]
  end

  test "test outer" do
    v1 = Numexy.new([1, 2, 3, 4])
    v2 = Numexy.new([4, 3, 2, 1])
    m = Numexy.outer(v1, v2)
    assert m.array == [[4, 3, 2, 1], [8, 6, 4, 2], [12, 9, 6, 3], [16, 12, 8, 4]]
    assert m.shape == [4, 4]

    m1 = Numexy.new([[1, 2], [3, 4]])
    m2 = Numexy.new([[4, 3], [2, 1]])
    m = Numexy.outer(m1, m2)
    assert m.array == [[4, 3, 2, 1], [8, 6, 4, 2], [12, 9, 6, 3], [16, 12, 8, 4]]
    assert m.shape == [4, 4]
  end
end
