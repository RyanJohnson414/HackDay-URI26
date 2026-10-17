import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import NetworkView from "./components/NetworkView";

const Page = window.location.pathname.replace(/\/$/, "") === "/network" ? NetworkView : App;

ReactDOM.createRoot(document.getElementById("root")!).render(
  React.createElement(React.StrictMode, null, React.createElement(Page)),
);
