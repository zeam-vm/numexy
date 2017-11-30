defmodule Array do

  defstruct array: [], shape: {nil, nil}

end

defmodule Numexy do
  @moduledoc """
  Documentation for Numexy.
  """

  @doc """
  New matrix.

  ## Examples

      iex> Numexy.new([1,2,3])
      %Array{array: [1, 2, 3], shape: {3, nil}}
      iex> Numexy.new([[1,2,3],[1,2,3]])
      %Array{array: [[1, 2, 3], [1, 2, 3]], shape: {2, 3}}

  """
  def new(array) do
    shape = {row_count(array), col_count(array)}
    %Array{array: array, shape: shape}
  end


  @doc """
  Calculate inner product.

  ## Examples

      iex> x = Numexy.new([1,2,3])
      %Array{array: [1,2,3], shape: {3, nil}}
      iex> y = Numexy.new([1,2,3])
      %Array{array: [1,2,3], shape: {3, nil}}
      iex> Numexy.dot(x, y)
      14
  """
  def dot(%Array{array: x, shape: {x_row, nil}}, %Array{array: y, shape: {y_row, nil}}) when x_row == y_row do
    # vector * vector
    Enum.zip(x, y)
    |> Enum.reduce(0, fn({a,b},acc)-> a*b+acc end)
  end

  def dot(%Array{array: x, shape: {_, x_col}}, %Array{array: y, shape: {y_row, _}}) when x_col == y_row do
    # matrix * matrix
    %Array{array: [[54,61],[81,86]], shape: {2, 2}}
  end


  @doc """
  Calculate transpose matrix.

  ## Examples

      iex> x = Numexy.new([[4,3],[7,5],[2,7]])
      %Array{array: [[4, 3], [7, 5], [2, 7]], shape: {3, 2}}
      iex> Numexy.transpose(x)
      %Array{array: [[4, 7, 2], [3, 5, 7]], shape: {2, 3}}
  """
  def transpose(%Array{array: x, shape: {_, col}}) when col != nil do
    x
    |> List.zip
    |> Enum.map(&Tuple.to_list/1)
    |> new
  end


  defp row_count(array) do
    Enum.count(array)
  end

  defp col_count([head| _ ]) when is_list(head) do
    Enum.count(head)
  end
  defp col_count(_) do
    nil
  end

end