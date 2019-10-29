defmodule Numexy do
  alias Numexy.Array

  @moduledoc """
  Documentation for Numexy.
  """

  @doc """
  New matrix.

  ## Examples

      iex> Numexy.new([1,2,3])
      %Numexy.Array{array: [1, 2, 3], shape: [3]}
      iex> Numexy.new([[1,2,3],[1,2,3]])
      %Numexy.Array{array: [[1, 2, 3], [1, 2, 3]], shape: [2, 3]}

  """
  def new(array), do: %Array{array: array, shape: count_list(array)}

  defp count_list(array) when is_list(array) do
    [Enum.count(array) | count_list(hd(array))]
  end

  defp count_list(_) do
    []
  end

  @doc """
  Add vector or matrix.

  ## Examples

      iex> Numexy.add(1, 2)
      3

      iex> Numexy.add([1, 2])
      3

      iex> x = Numexy.new([1,2,3])
      %Numexy.Array{array: [1,2,3], shape: [3]}
      iex> y = 4
      iex> Numexy.add(x, y)
      %Numexy.Array{array: [5,6,7], shape: [3]}

      iex> x = Numexy.new([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
      %Numexy.Array{
              array: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
              shape: [3, 2, 2]
            }
      iex> y = 4
      iex> Numexy.add(x, y)
      %Numexy.Array{
              array: [[[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]],
              shape: [3, 2, 2]
            }

  """
  def add(s, t) when is_number(s) and is_number(t), do: s + t

  def add(%Array{array: v, shape: shape}, s) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(&1 + s)) |> chunk(tl(shape)) |> new

  def add(s, %Array{array: v, shape: shape}) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(s + &1)) |> chunk(tl(shape)) |> new

  def add(%Array{array: x, shape: shape}, %Array{array: y, shape: shape}) do
    Enum.zip(List.flatten(x), List.flatten(y))
    |> Enum.map(fn {a, b} -> a + b end)
    |> chunk(tl(shape))
    |> new
  end

  def add(l) when is_list(l) do
    Enum.reduce(l, fn x, acc -> add(acc, x) end)
  end

  defp chunk(list, []), do: list

  defp chunk(list, [head | tail]) do
    Enum.chunk_every(chunk(list, tail), head)
  end

  @doc """
  Subtraction vector or matrix.

  ## Examples

      iex> x = Numexy.new([1,2,3])
      %Numexy.Array{array: [1,2,3], shape: [3]}
      iex> y = 4
      iex> Numexy.sub(x, y)
      %Numexy.Array{array: [-3,-2,-1], shape: [3]}
  """
  def sub(s, t) when is_number(s) and is_number(t), do: s - t

  def sub(%Array{array: v, shape: shape}, s) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(&1 - s)) |> chunk(tl(shape)) |> new

  def sub(s, %Array{array: v, shape: shape}) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(s - &1)) |> chunk(tl(shape)) |> new

  def sub(%Array{array: x, shape: shape}, %Array{array: y, shape: shape}) do
    Enum.zip(List.flatten(x), List.flatten(y))
    |> Enum.map(fn {a, b} -> a - b end)
    |> chunk(tl(shape))
    |> new
  end

  def sub(l) when is_list(l) do
    Enum.reduce(l, fn x, acc -> sub(acc, x) end)
  end

  @doc """
  Multiplication vector or matrix.

  ## Examples

      iex> x = Numexy.new([1,2,3])
      %Numexy.Array{array: [1,2,3], shape: [3]}
      iex> y = 4
      iex> Numexy.mul(x, y)
      %Numexy.Array{array: [4,8,12], shape: [3]}
  """
  def mul(s, t) when is_number(s) and is_number(t), do: s * t

  def mul(%Array{array: v, shape: shape}, s) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(&1 * s)) |> chunk(tl(shape)) |> new

  def mul(s, %Array{array: v, shape: shape}) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(s * &1)) |> chunk(tl(shape)) |> new

  def mul(%Array{array: x, shape: shape}, %Array{array: y, shape: shape}) do
    Enum.zip(List.flatten(x), List.flatten(y))
    |> Enum.map(fn {a, b} -> a * b end)
    |> chunk(tl(shape))
    |> new
  end

  def mul(l) when is_list(l) do
    Enum.reduce(l, fn x, acc -> mul(acc, x) end)
  end

  @doc """
  Division vector or matrix.

  ## Examples

      iex> x = Numexy.new([8,4,2])
      %Numexy.Array{array: [8,4,2], shape: [3]}
      iex> y = 4
      iex> Numexy.div(x, y)
      %Numexy.Array{array: [2.0,1.0,0.5], shape: [3]}
  """
  def div(s, t) when is_number(s) and is_number(t), do: s / t

  def div(%Array{array: v, shape: shape}, s) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(&1 / s)) |> chunk(tl(shape)) |> new

  def div(s, %Array{array: v, shape: shape}) when is_number(s),
    do: v |> List.flatten() |> Enum.map(&(s / &1)) |> chunk(tl(shape)) |> new

  def div(%Array{array: x, shape: shape}, %Array{array: y, shape: shape}) do
    Enum.zip(List.flatten(x), List.flatten(y))
    |> Enum.map(fn {a, b} -> a / b end)
    |> chunk(tl(shape))
    |> new
  end

  def div(l) when is_list(l) do
    Enum.reduce(l, fn x, acc -> Numexy.div(acc, x) end)
  end

  @doc """
  Calculate inner product.

  ## Examples

      iex> x = Numexy.new([1,2,3])
      %Numexy.Array{array: [1,2,3], shape: [3]}
      iex> y = Numexy.new([1,2,3])
      %Numexy.Array{array: [1,2,3], shape: [3]}
      iex> Numexy.dot(x, y)
      14
  """
  def dot(%Array{array: xv, shape: [row]}, %Array{array: yv, shape: [row]}) do
    # vector * vector return scalar
    dot_vector(xv, yv)
  end

  def dot(%Array{array: m, shape: [_, mv]}, %Array{array: v, shape: [mv]}) do
    # matrix * vector return vector
    m = for mi <- m, vi <- [v], do: [mi, vi]

    m
    |> Enum.map(fn [x, y] -> dot_vector(x, y) end)
    |> new
  end

  def dot(%Array{array: xm, shape: [x_row, xy]}, %Array{array: ym, shape: [xy, _]}) do
    # matrix * matrix return matrix
    m = for xi <- xm, yi <- list_transpose(ym), do: [xi, yi]

    m
    |> Enum.map(fn [x, y] -> dot_vector(x, y) end)
    |> Enum.chunk_every(x_row)
    |> new
  end

  defp dot_vector(xv, yv) do
    Enum.zip(xv, yv)
    |> Enum.reduce(0, fn {a, b}, acc -> a * b + acc end)
  end

  @doc """
  Calculate transpose matrix.

  ## Examples

      iex> x = Numexy.new([[4,3],[7,5],[2,7]])
      %Array{array: [[4, 3], [7, 5], [2, 7]], shape: [3, 2]}
      iex> Numexy.transpose(x)
      %Array{array: [[4, 7, 2], [3, 5, 7]], shape: [2, 3]}
  """
  def transpose(%Array{array: m, shape: [_, _]}) do
    m
    |> list_transpose
    |> new
  end

  defp list_transpose(list) do
    list
    |> List.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  @doc """
  Create ones matrix or vector.

  ## Examples

      iex> Numexy.ones([2, 3])
      %Array{array: [[1, 1, 1], [1, 1, 1]], shape: [2, 3]}
      iex> Numexy.ones([3])
      %Array{array: [1, 1, 1], shape: [3]}
      iex> Numexy.ones({2, 3})
      %Array{array: [[1, 1, 1], [1, 1, 1]], shape: [2, 3]}
      iex> Numexy.ones({3, nil})
      %Array{array: [1, 1, 1], shape: [3]}
  """
  def ones([row]) do
    List.duplicate(1, row)
    |> new
  end

  def ones([row, col]) do
    List.duplicate(1, col)
    |> List.duplicate(row)
    |> new
  end

  def ones({row, nil}), do: ones([row])
  def ones({row, col}), do: ones([row, col])

  @doc """
  Create zeros matrix or vector.

  ## Examples

      iex> Numexy.zeros([2, 3])
      %Numexy.Array{array: [[0, 0, 0], [0, 0, 0]], shape: [2, 3]}
      iex> Numexy.zeros([3])
      %Numexy.Array{array: [0, 0, 0], shape: [3]}
      iex> Numexy.zeros({2, 3})
      %Numexy.Array{array: [[0, 0, 0], [0, 0, 0]], shape: [2, 3]}
      iex> Numexy.zeros({3, nil})
      %Numexy.Array{array: [0, 0, 0], shape: [3]}
  """
  def zeros([row]) do
    List.duplicate(0, row)
    |> new
  end

  def zeros([row, col]) do
    List.duplicate(0, col)
    |> List.duplicate(row)
    |> new
  end

  def zeros({row, nil}), do: zeros([row])
  def zeros({row, col}), do: zeros([row, col])

  @doc """
  Sum matrix or vector.

  ## Examples

      iex> Numexy.new([2,9,5]) |> Numexy.sum
      16
      iex> Numexy.new([[1,2,3],[4,5,6]]) |> Numexy.sum
      21
  """
  def sum(%Array{array: v, shape: [_]}) do
    v
    |> Enum.reduce(&(&1 + &2))
  end

  def sum(%Array{array: m, shape: _}) do
    m
    |> Enum.reduce(0, &(Enum.reduce(&1, fn x, acc -> x + acc end) + &2))
  end

  @doc """
  Avarage matrix or vector.

  ## Examples

      iex> Numexy.new([2,9,5]) |> Numexy.avg
      5.333333333333333
      iex> Numexy.new([[1,2,3],[4,5,6]]) |> Numexy.avg
      3.5
  """
  def avg(%Array{array: v, shape: [row]}) do
    v
    |> Enum.reduce(&(&1 + &2))
    |> float_div(row)
  end

  def avg(%Array{array: m, shape: [row, col]}) do
    m
    |> Enum.reduce(0, &(Enum.reduce(&1, fn x, acc -> x + acc end) + &2))
    |> float_div(row * col)
  end

  defp float_div(dividend, divisor) do
    dividend / divisor
  end

  @doc """
  Get matrix or vector value.

  ## Examples

      iex> Numexy.new([2,9,5]) |> Numexy.get([2])
      9
      iex> Numexy.new([[1,2,3],[4,5,6]]) |> Numexy.get([2, 1])
      4
      iex> Numexy.new([2,9,5]) |> Numexy.get({2, nil})
      9
      iex> Numexy.new([[1,2,3],[4,5,6]]) |> Numexy.get({2, 1})
      4
  """
  def get(%Array{array: v, shape: [_]}, [row]), do: Enum.at(v, row - 1)
  def get(%Array{array: m, shape: _}, [row, col]), do: Enum.at(m, row - 1) |> Enum.at(col - 1)
  def get(%Array{array: v, shape: [_]}, {row, nil}), do: Enum.at(v, row - 1)
  def get(%Array{array: m, shape: _}, {row, col}), do: Enum.at(m, row - 1) |> Enum.at(col - 1)

  @doc """
  Get index of max value.

  ## Examples

      iex> Numexy.new([[1,2,9],[4,5,6]]) |> Numexy.argmax
      2
  """
  def argmax(%Array{array: v, shape: [_]}), do: v |> find_max_value_index

  def argmax(%Array{array: m, shape: _}) do
    m |> find_max_value_index
  end

  @doc """
  Get index of max value row or col.

  ## Examples

      iex> Numexy.new([[1,2,9],[4,6,3]]) |> Numexy.argmax(:row)
      [2, 1]
      iex> Numexy.new([[1,2,9],[4,6,3]]) |> Numexy.argmax(:col)
      [1, 1, 0]
  """
  def argmax(%Array{array: m, shape: _}, :row) do
    m
    |> Enum.map(&find_max_value_index(&1))
  end

  def argmax(%Array{array: m, shape: _}, :col) do
    m
    |> list_transpose
    |> Enum.map(&find_max_value_index(&1))
  end

  defp find_max_value_index(list) do
    flat_list = List.flatten(list)
    max_value = Enum.max(flat_list)
    flat_list |> Enum.find_index(&(&1 == max_value))
  end

  @doc """
  Get step function value.

  ## Examples

      iex> Numexy.new([-2,9,5]) |> Numexy.step_function()
      %Numexy.Array{array: [0, 1, 1], shape: [3]}
  """
  def step_function(%Array{array: v, shape: [_]}) do
    v
    |> Enum.map(&step_function_output(&1))
    |> new
  end

  defp step_function_output(num) when num > 0, do: 1
  defp step_function_output(num) when num <= 0, do: 0

  @doc """
  Get sigmoid function value.

  ## Examples

      iex> Numexy.new([-2,9,5]) |> Numexy.sigmoid()
      %Numexy.Array{array: [0.11920292202211755, 0.9998766054240137, 0.9933071490757153], shape: [3]}
  """
  def sigmoid(%Array{array: v, shape: [_]}) do
    v
    |> Enum.map(&(1 / (1 + :math.exp(-1 * &1))))
    |> new
  end

  @doc """
  Get relu function value.

  ## Examples

      iex> Numexy.new([-2,9,5]) |> Numexy.relu()
      %Numexy.Array{array: [0, 9, 5], shape: [3]}
  """
  def relu(%Array{array: v, shape: [_]}) do
    v
    |> Enum.map(&relu_output(&1))
    |> new
  end

  defp relu_output(x) when x > 0, do: x
  defp relu_output(x) when x <= 0, do: 0

  @doc """
  Get softmax function value.

  ## Examples

      iex> Numexy.new([-2,9,5]) |> Numexy.softmax()
      %Numexy.Array{array: [1.6401031494862326e-5, 0.9819976839988096, 0.017985914969695496], shape: [3]}
  """
  def softmax(%Array{array: v, shape: [_]}) do
    sum_num = Enum.reduce(v, 0, &(:math.exp(&1) + &2))

    v
    |> Enum.map(&(:math.exp(&1) / sum_num))
    |> new
  end

  @doc """
  Reshape list.

  ## Examples

      iex> Numexy.reshape([1,2,3,4,5,6], 3)
      %Numexy.Array{array: [[1,2,3],[4,5,6]], shape: [2, 3]}
  """
  def reshape(list, col) when rem(length(list), col) == 0 do
    list
    |> Enum.chunk_every(col)
    |> new
  end

  @doc """
  Calculate outer product.

  ## Examples

      iex> Numexy.new([1,2,3,4]) |> Numexy.outer(Numexy.new([4,3,2,1]))
      %Numexy.Array{array: [[4,3,2,1],[8,6,4,2],[12,9,6,3],[16,12,8,4]], shape: [4, 4]}

  """
  def outer(%Array{array: array1, shape: _}, %Array{array: array2, shape: _}) do
    list1 = List.flatten(array1)
    list2 = List.flatten(array2)

    Enum.map(list1, &Enum.map(list2, fn x -> x * &1 end))
    |> new
  end
end
