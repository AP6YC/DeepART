using Plots
unicodeplots()
using ProgressMeter
# using UnicodePlots

n = 20
x = collect(1:n)
y = sqrt.(x)

t_pause = 0.1

pro = Progress(n)
# io = open("/dev/pts/3", "w")
# ioc = IOContext(io, :color => true)
# pro = Progress(n; output=ioc)

generate_showvalues(p) = () -> [
    (:p, p)
    # (:p, show(p)),
    # (:p, string(p)),
    # (:a, a)
]

erase_to_end_of_line(output_stream::IO) = print(output_stream, "\033[K")
move_up_1_line(output_stream::IO) = print(output_stream, "\033[1A")
move_down_1_line(output_stream::IO) = print(output_stream, "\033[1B")
go_to_start_of_line(output_stream::IO) = print(output_stream, "\r")

for ix in 1:n
    # local_plot = lineplot(
    local_plot = plot(
        x[1:ix],
        y[1:ix],
        # xlims=(0, 10),
        # ylims=(0, 4),
        title="Square root",
        xlabel="x",
        ylabel="y",
        height=15,
    )
    # if ix > 1
    #     # erase_to_end_of_line(stdout)
    #     go_to_start_of_line(stdout)
    #     for jx = 1:18
    #         move_up_1_line(stdout)
    #     end
    # end
    # print(show(p))
    # print(stdout, show(p))
    # show(p)

    # a = ""
    # if ix < n
    #     a *= "\r"
    #     for jx = 1:19
    #     # for jx = 1:10
    #         a *= "\033[1A"
    #     end
    #     a *= "\n"
    # end
    # a *= string(show(p))

    str = ""
    # str *= "\n"
    # # str *= "\r"
    # for jx = 1:8
    #     str *= "\033[1A"
    # end
    # str *= "\n" * string(show(local_plot)) # use ANSI color codes and prepend newline
    # str = string(p; color=true) # use ANSI color codes and prepend newline
    str = "\n" * string(show(local_plot)) * "\n"

    # next!(pro; showvalues=generate_showvalues(str))
    next!(pro;
        showvalues = [
            # (:UnicodePlot, replace_EOL_with_space(str))
            (:UnicodePlot, str)
        ]
    )
    sleep(t_pause)
    # a = savefig(local_plot, "testplot.txt")
end

# n = 20
# xs = Float64[]
# p = Progress(n)
# for iter = 1:n
#     append!(xs, rand())
#     sleep(0.5)
#     plot = lineplot(xs)
#     str = "\n" * string(plot; color=true) # use ANSI color codes and prepend newline
#     ProgressMeter.next!(p; showvalues = [(:UnicodePlot, str)])
# end

# function replace_EOL_with_space(s)
#     width = displaysize(stdout)[2]
#     lines = split(s, "\n")
#     filled = lines .* " ".^(width .- sizeof.(lines))
#     return join(filled)
# end

# function testplot(n = 10)
#     p = Progress(n)
#     for iter in 1:n
#         sleep(0.2)
#         s = "line 1\nline 2\nline 3"
#         next!(p; showvalues = [("lines", replace_EOL_with_space(s))])
#     end
# end

# testplot()
