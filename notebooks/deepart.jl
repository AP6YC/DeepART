### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 4c9c839d-de70-44b8-b5e9-ec56dbde1918
begin
	cd(joinpath(@__DIR__, ".."))
	using Pkg
	Pkg.activate(".")
	@info "Local DeepART project activated"
end

# ╔═╡ 6b2ec3f0-a4ef-11ee-198f-e7efe4d5ed49
begin
	using Revise
	using PlutoUI
	using DeepART
	using DataStructures
	include(joinpath("src", "pluto", "PlutoUtils.jl"))
	using .PlutoUtils
	@info "Dependencies loaded"
end

# ╔═╡ 90927cb7-6d97-46a1-8e3b-37a3cce394b1
md"""
# DeepART Project Drafting Notebook
"""

# ╔═╡ 9e191ddb-6915-4d90-b214-340702bd3acd
md"""
## Load Dependencies
"""

# ╔═╡ 8d637d62-d5c0-4de9-a876-0d0dcac4aaee
TableOfContents(title="Experiments 🔬")

# ╔═╡ 9354d864-7504-47ee-b5c0-e34a8fac75c3
md"## Experiments"

# ╔═╡ 33ceb4c7-22fb-4ea9-9c05-7a37975e8e7f
config_dict = OrderedDict(
	"n_train" => PlutoUtils.NumConfig(1, 1000, 500),
	"n_test" => PlutoUtils.NumConfig(1, 1000, 200),
	"model_bools" => PlutoUtils.OptConfig(["dense", "second_conv"]),
	"rho_lb" => PlutoUtils.NumConfig(0.01, 0.99, 0.2, 0.01),
	"rho_ub" => PlutoUtils.NumConfig(0.01, 0.99, 0.99, 0.01),
)

# ╔═╡ b2ff33a2-f746-4228-8787-d0f52d4d9528
@bind config confirm(PlutoUtils.config_input(config_dict))

# ╔═╡ 084cb7df-5c4d-432f-baa5-7db92cceefd6
# DeepART.inspect_prediction_errors(data.test, y_hat, selected, n_show)

# ╔═╡ c8b4db44-dcce-48c7-a327-8ffcc528b574
md"""
### Load Data
"""

# ╔═╡ b7a0014f-5155-4350-9e1d-b3f77e2d256e
data = DeepART.get_mnist()

# ╔═╡ 6a536ecd-b256-4fc1-ae38-d0bf5a1dd6bd
DeepART.inspect_truth_errors(data.test, y_hat, selected, n_show)

# ╔═╡ aee8d6c2-4c69-4b7a-a458-5a91f8f44536
DeepART.view_filts(convart.model)

# ╔═╡ feef3708-184f-413c-a801-d02f2fa2f5fd
# DeepART.tryit()

# ╔═╡ e376f339-ea0e-452f-8a86-ac2cb43f6a37
md"### 1: Cats and Dogs"

# ╔═╡ c5d75792-2d7c-4c6b-abea-c1ec48c73a95
begin
	dog_slider = @bind 🐶 Slider(1:10, default=5, show_value=true)
	cat_slider = @bind 🐱 Slider(11:20, default=12, show_value=true)

	md"""
	**How many pets do you have?**

	Dogs 🐶: $(dog_slider)

	Cats 😺: $(cat_slider)
	"""
end

# ╔═╡ 6234a9b7-a17a-412b-996f-7a70c4d3274c
md"
You have $🐶 dogs and $(🐱) cats
"

# ╔═╡ 166942a5-8943-4bda-b2d5-92deca39b70a
md"### 2. Meaning of Life"

# ╔═╡ bfb45c94-752f-4c08-b096-9940bcbbd603
md"*What is the meaning of life?*

$(@bind x Slider(1:42, default=31, show_value=true))
"

# ╔═╡ b4ba95e9-06d2-40a2-985a-5540cfd6678a
if x == 42
	PlutoUtils.correct(md"YOU HAVE FOUND THE ANSWER")
elseif 30 < x < 42
	PlutoUtils.almost(md"YOU HAVE ALMOST FOUND THE ANSWER")
else
	PlutoUtils.keep_working(md"THAT IS NOT THE ANSWER")
end

# ╔═╡ 621a4454-d76b-4d92-9c2d-7b88a513bb1a
PlutoUtils.hint(md"Don't forget to bring a towel")

# ╔═╡ Cell order:
# ╟─90927cb7-6d97-46a1-8e3b-37a3cce394b1
# ╟─9e191ddb-6915-4d90-b214-340702bd3acd
# ╟─4c9c839d-de70-44b8-b5e9-ec56dbde1918
# ╟─6b2ec3f0-a4ef-11ee-198f-e7efe4d5ed49
# ╠═8d637d62-d5c0-4de9-a876-0d0dcac4aaee
# ╟─9354d864-7504-47ee-b5c0-e34a8fac75c3
# ╟─33ceb4c7-22fb-4ea9-9c05-7a37975e8e7f
# ╟─b2ff33a2-f746-4228-8787-d0f52d4d9528
# ╟─6a536ecd-b256-4fc1-ae38-d0bf5a1dd6bd
# ╠═084cb7df-5c4d-432f-baa5-7db92cceefd6
# ╟─c8b4db44-dcce-48c7-a327-8ffcc528b574
# ╟─b7a0014f-5155-4350-9e1d-b3f77e2d256e
# ╠═aee8d6c2-4c69-4b7a-a458-5a91f8f44536
# ╠═feef3708-184f-413c-a801-d02f2fa2f5fd
# ╟─e376f339-ea0e-452f-8a86-ac2cb43f6a37
# ╟─c5d75792-2d7c-4c6b-abea-c1ec48c73a95
# ╟─6234a9b7-a17a-412b-996f-7a70c4d3274c
# ╟─166942a5-8943-4bda-b2d5-92deca39b70a
# ╟─bfb45c94-752f-4c08-b096-9940bcbbd603
# ╠═b4ba95e9-06d2-40a2-985a-5540cfd6678a
# ╟─621a4454-d76b-4d92-9c2d-7b88a513bb1a
