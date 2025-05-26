#pragma once

#include "CMakeLists.g.h"

namespace winrt::OPENCL::implementation
{
    struct CMakeLists : CMakeListsT<CMakeLists>
    {
        CMakeLists() 
        {
            // Xaml objects should not call InitializeComponent during construction.
            // See https://github.com/microsoft/cppwinrt/tree/master/nuget#initializecomponent
        }

        int32_t MyProperty();
        void MyProperty(int32_t value);

        void ClickHandler(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::RoutedEventArgs const& args);
    };
}

namespace winrt::OPENCL::factory_implementation
{
    struct CMakeLists : CMakeListsT<CMakeLists, implementation::CMakeLists>
    {
    };
}
