import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { HealthProvider } from "./hooks/HealthContext";
import { AboutPage } from "./pages/AboutPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { DashboardPage } from "./pages/DashboardPage";
import { HistoryPage } from "./pages/HistoryPage";

export default function App() {
  return (
    <HealthProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<AnalyzePage />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="history/:id" element={<HistoryPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HealthProvider>
  );
}
