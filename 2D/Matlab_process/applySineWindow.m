function windowed_data = applySineWindow(data, off, endVal, pow, c)
    
    N = length(data);

    % Generate the Sine window function
    t = linspace(off, endVal, N);
    window = sin(pi * t.^pow);
    % Apply the first-point scale factor
    window(1) = window(1) * c;

    % Apply the window function to the data
    windowed_data = data.* window;
  
end