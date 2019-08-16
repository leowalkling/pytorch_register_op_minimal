#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include <torch/script.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <sstream>

namespace py = pybind11;
namespace ta = torch::autograd;
using std::int64_t;
using std::shared_ptr;
using std::make_shared;
using torch::autograd::Variable;
using torch::autograd::variable_list;
using at::Tensor;
using std::ostringstream;


struct IdentityBackward : public torch::autograd::TraceableFunction {
	using TraceableFunction::TraceableFunction;
	torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;
	std::string name() const override { return "IdentityBackward"; }
	void release_variables() override {
	}
};


variable_list IdentityBackward::apply(variable_list&& grads) {
	auto num_outputs{ this->num_outputs() };

	TORCH_CHECK(grads.size() == num_outputs,
		static_cast<ostringstream&>(ostringstream{} << "Expected " << num_outputs << " output gradients in TPIdentityBackward but got " << grads.size() << "!").str());
	variable_list grads_in{ static_cast<size_t>(num_outputs) };
	for (auto i{ 0U }; i < num_outputs; ++i) {
		if (should_compute_output(i)) {
			grads_in[i] = std::move(grads[i]);
		}
	}
	return std::move(grads_in);
}

at::Tensor new_view(const at::Tensor & t) {
	auto base_var = Variable(t);
	if (base_var.is_view()) {
		base_var = base_var.base();
	}
	auto r{
		torch::autograd::make_variable_view(base_var, base_var.tensor_data(), true)
	};

	shared_ptr<IdentityBackward> grad_fn;
	if (ta::compute_requires_grad(t)) {
		grad_fn = shared_ptr<IdentityBackward>(new IdentityBackward());
		grad_fn->set_next_edges(ta::collect_next_edges(t));
		ta::set_history(r, grad_fn);
	}

	if (torch::jit::tracer::isTracing()) {
		TORCH_CHECK(0, "tracing not supported");
	}

	return std::move(r);
}

static auto ops{
	torch::RegisterOperators()
	.op("myops::new_view(Tensor(a) t) -> Tensor(a)", &new_view)
};
