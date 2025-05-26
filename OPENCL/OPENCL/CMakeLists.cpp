#include "pch.h"
#include "CMakeLists.h"
#if __has_include("CMakeLists.g.cpp")
#include "CMakeLists.g.cpp"
#endif

using namespace winrt;
using namespace Windows::UI::Xaml;

namespace winrt::OPENCL::implementation
{
    int32_t CMakeLists::MyProperty()
    {
        throw hresult_not_implemented();
    }

    void CMakeLists::MyProperty(int32_t /* value */)
    {
        throw hresult_not_implemented();
    }

    void CMakeLists::ClickHandler(IInspectable const&, RoutedEventArgs const&)
    {
        Button().Content(box_value(L"Clicked"));
    }
}
